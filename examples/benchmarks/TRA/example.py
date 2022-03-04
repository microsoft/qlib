import argparse

import qlib
import ruamel.yaml as yaml
from qlib.utils import init_instance_by_config


def main(seed, config_file="configs/config_alstm.yaml"):

    # set random seed
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # seed_suffix = "/seed1000" if "init" in config_file else f"/seed{seed}"
    seed_suffix = ""
    config["task"]["model"]["kwargs"].update(
        {"seed": seed, "logdir": config["task"]["model"]["kwargs"]["logdir"] + seed_suffix}
    )

    # initialize workflow
    qlib.init(
        provider_uri=config["qlib_init"]["provider_uri"],
        region=config["qlib_init"]["region"],
    )
    dataset = init_instance_by_config(config["task"]["dataset"])
    model = init_instance_by_config(config["task"]["model"])

    # train model
    model.fit(dataset)


if __name__ == "__main__":

    # set params from cmd
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--seed", type=int, default=1000, help="random seed")
    parser.add_argument("--config_file", type=str, default="configs/config_alstm.yaml", help="config file")
    args = parser.parse_args()
    main(**vars(args))
