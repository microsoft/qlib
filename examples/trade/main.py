import re
import os
import argparse
import yaml
from executor import Executor
import warnings
import redis
import subprocess

warnings.filterwarnings("ignore")

from util import merge_dicts

loader = yaml.FullLoader
loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def get_full_config(config, dir_name):
    while "base" in config:
        base_config = os.path.normpath(os.path.join(dir_name, config.pop("base")))
        dir_name = os.path.dirname(base_config)
        with open(base_config, "r") as f:
            base_config = yaml.load(base_config, Loader=yaml.FullLoader)
        config = merge_dicts(base_config, config)
    return config


def run(config):
    log_dir = config["log_dir"]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(log_dir + "/config.yml", "w") as f:
        yaml.dump(config, f)
    executor = Executor(**config)
    if config["task"] == "train":
        return executor.train(**config["optim"])
    elif config["task"] == "eval":
        return executor.eval(config["test_paths"]["order_dir"], save_res=True, logdir=config["log_dir"] + "/test/",)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str)
    parser.add_argument("-n", "--index", type=int, default=None)
    args = parser.parse_args()

    print(os.cpu_count())

    EXP_PATH = os.environ["EXP_PATH"]
    config_path = os.path.normpath(os.path.join(EXP_PATH, args.config))
    EXP_NAME = os.path.relpath(config_path, EXP_PATH)
    if os.path.isdir(config_path):
        if not args.index is None:
            with open(config_path + "/configs.yml") as f:
                config_list = list(yaml.load_all(f, Loader=loader))
            config = config_list[args.index]
            if "PT_OUTPUT_DIR" in os.environ:
                config["log_dir"] = os.environ["PT_OUTPUT_DIR"]
            else:
                log_prefix = os.environ["OUTPUT_DIR"] if "OUTPUT_DIR" in os.environ else "../log"
                config["log_dir"] = os.path.join(log_prefix, config["log_dir"])
            config = get_full_config(config, config_path)
            run(config)
        else:
            redis_server = redis.Redis(
                host=os.environ["REDIS_SERVER"],
                port=os.environ["REDIS_PORT"],
                db=0,
                charset="utf-8",
                decode_responses=True,
            )
            with open(config_path + "/configs.yml") as f:
                config_list = list(yaml.load_all(f, Loader=loader))
            config_num = len(config_list)
            if not redis_server.exists(EXP_NAME):
                for i in range(config_num):
                    redis_server.rpush(EXP_NAME, i)
                    redis_server.set(f"{EXP_NAME}_{i}", "Pending")
            else:
                if redis_server.llen(EXP_NAME) == 0:
                    for i in range(config_num):
                        if (
                            not redis_server.exists(f"{EXP_NAME}_{i}")
                            or redis_server.get(f"{EXP_NAME}_{i}") == "Failed"
                        ):
                            redis_server.rpush(EXP_NAME, i)
                            redis_server.set(f"{EXP_NAME}_{i}", "Pending")
            print(f"Starting..., {redis_server.llen(EXP_NAME)} trails to run")
            while True:
                index = redis_server.lpop(EXP_NAME)
                if index is None:
                    print("All done")
                    break
                index = int(index)
                redis_server.set(f"{EXP_NAME}_{index}", "Running")
                print(f"Trail_{index} is running")
                try:
                    res = subprocess.run(["python", "main.py", "--config", args.config, "--index", str(index),],)
                except KeyboardInterrupt:
                    redis_server.set(f"{EXP_NAME}_{index}", "Failed")
                    print(f"Trail_{index} has failed, {redis_server.llen(EXP_NAME)} trails to run")
                    break
                if res.returncode == 0:
                    redis_server.set(f"{EXP_NAME}_{index}", "Finished")
                    print(f"Finish running one trail, {redis_server.llen(EXP_NAME)} trails to run")
                else:
                    redis_server.set(f"{EXP_NAME}_{index}", "Failed")
                    print(f"Trail_{index} has failed, {redis_server.llen(EXP_NAME)} trails to run")

    elif os.path.isfile(config_path):
        assert config_path.endswith(".yml"), "Config file should be an yaml file"
        EXP_NAME = EXP_NAME[:-4]
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=loader)
        config = get_full_config(config, os.path.dirname(config_path))
        log_prefix = os.environ["OUTPUT_DIR"] if "OUTPUT_DIR" in os.environ else "../log"
        config["log_dir"] = os.path.join(log_prefix, config["log_dir"])
        run(config)
    else:
        print("The config path should be a relative path from EXP_PATH")
