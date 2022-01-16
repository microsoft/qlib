#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import os
import sys
import fire
import time
import glob
import yaml
import shutil
import signal
import inspect
import tempfile
import functools
import statistics
import subprocess
from datetime import datetime
from pathlib import Path
from operator import xor
from pprint import pprint

import qlib
from qlib.workflow import R
from qlib.tests.data import GetData


# decorator to check the arguments
def only_allow_defined_args(function_to_decorate):
    @functools.wraps(function_to_decorate)
    def _return_wrapped(*args, **kwargs):
        """Internal wrapper function."""
        argspec = inspect.getfullargspec(function_to_decorate)
        valid_names = set(argspec.args + argspec.kwonlyargs)
        if "self" in valid_names:
            valid_names.remove("self")
        for arg_name in kwargs:
            if arg_name not in valid_names:
                raise ValueError("Unknown argument seen '%s', expected: [%s]" % (arg_name, ", ".join(valid_names)))
        return function_to_decorate(*args, **kwargs)

    return _return_wrapped


# function to handle ctrl z and ctrl c
def handler(signum, frame):
    os.system("kill -9 %d" % os.getpid())


signal.signal(signal.SIGINT, handler)


# function to calculate the mean and std of a list in the results dictionary
def cal_mean_std(results) -> dict:
    mean_std = dict()
    for fn in results:
        mean_std[fn] = dict()
        for metric in results[fn]:
            mean = statistics.mean(results[fn][metric]) if len(results[fn][metric]) > 1 else results[fn][metric][0]
            std = statistics.stdev(results[fn][metric]) if len(results[fn][metric]) > 1 else 0
            mean_std[fn][metric] = [mean, std]
    return mean_std


# function to create the environment ofr an anaconda environment
def create_env():
    # create env
    temp_dir = tempfile.mkdtemp()
    env_path = Path(temp_dir).absolute()
    sys.stderr.write(f"Creating Virtual Environment with path: {env_path}...\n")
    execute(f"conda create --prefix {env_path} python=3.7 -y")
    python_path = env_path / "bin" / "python"  # TODO: FIX ME!
    sys.stderr.write("\n")
    # get anaconda activate path
    conda_activate = Path(os.environ["CONDA_PREFIX"]) / "bin" / "activate"  # TODO: FIX ME!
    return temp_dir, env_path, python_path, conda_activate


# function to execute the cmd
def execute(cmd, wait_when_err=False, raise_err=True):
    print("Running CMD:", cmd)
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True, shell=True) as p:
        for line in p.stdout:
            sys.stdout.write(line.split("\b")[0])
            if "\b" in line:
                sys.stdout.flush()
                time.sleep(0.1)
                sys.stdout.write("\b" * 10 + "\b".join(line.split("\b")[1:-1]))

    if p.returncode != 0:
        if wait_when_err:
            input("Press Enter to Continue")
        if raise_err:
            raise RuntimeError(f"Error when executing command: {cmd}")
        return p.stderr
    else:
        return None


# function to get all the folders benchmark folder
def get_all_folders(models, exclude) -> dict:
    folders = dict()
    if isinstance(models, str):
        model_list = models.split(",")
        models = [m.lower().strip("[ ]") for m in model_list]
    elif isinstance(models, list):
        models = [m.lower() for m in models]
    elif models is None:
        models = [f.name.lower() for f in os.scandir("benchmarks")]
    else:
        raise ValueError("Input models type is not supported. Please provide str or list without space.")
    for f in os.scandir("benchmarks"):
        add = xor(bool(f.name.lower() in models), bool(exclude))
        if add:
            path = Path("benchmarks") / f.name
            folders[f.name] = str(path.resolve())
    return folders


# function to get all the files under the model folder
def get_all_files(folder_path, dataset) -> (str, str):
    yaml_path = str(Path(f"{folder_path}") / f"*{dataset}*.yaml")
    req_path = str(Path(f"{folder_path}") / f"*.txt")
    yaml_file = glob.glob(yaml_path)
    req_file = glob.glob(req_path)
    if len(yaml_file) == 0:
        return None, None
    else:
        return yaml_file[0], req_file[0]


# function to retrieve all the results
def get_all_results(folders) -> dict:
    results = dict()
    for fn in folders:
        try:
            exp = R.get_exp(experiment_name=fn, create=False)
        except ValueError:
            # No experiment results
            continue
        recorders = exp.list_recorders()
        result = dict()
        result["annualized_return_with_cost"] = list()
        result["information_ratio_with_cost"] = list()
        result["max_drawdown_with_cost"] = list()
        result["ic"] = list()
        result["icir"] = list()
        result["rank_ic"] = list()
        result["rank_icir"] = list()
        for recorder_id in recorders:
            if recorders[recorder_id].status == "FINISHED":
                recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=fn)
                metrics = recorder.list_metrics()
                if "1day.excess_return_with_cost.annualized_return" not in metrics:
                    print(f"{recorder_id} is skipped due to incomplete result")
                    continue
                result["annualized_return_with_cost"].append(metrics["1day.excess_return_with_cost.annualized_return"])
                result["information_ratio_with_cost"].append(metrics["1day.excess_return_with_cost.information_ratio"])
                result["max_drawdown_with_cost"].append(metrics["1day.excess_return_with_cost.max_drawdown"])
                result["ic"].append(metrics["IC"])
                result["icir"].append(metrics["ICIR"])
                result["rank_ic"].append(metrics["Rank IC"])
                result["rank_icir"].append(metrics["Rank ICIR"])
        results[fn] = result
    return results


# function to generate and save markdown table
def gen_and_save_md_table(metrics, dataset):
    table = "| Model Name | Dataset | IC | ICIR | Rank IC | Rank ICIR | Annualized Return | Information Ratio | Max Drawdown |\n"
    table += "|---|---|---|---|---|---|---|---|---|\n"
    for fn in metrics:
        ic = metrics[fn]["ic"]
        icir = metrics[fn]["icir"]
        ric = metrics[fn]["rank_ic"]
        ricir = metrics[fn]["rank_icir"]
        ar = metrics[fn]["annualized_return_with_cost"]
        ir = metrics[fn]["information_ratio_with_cost"]
        md = metrics[fn]["max_drawdown_with_cost"]
        table += f"| {fn} | {dataset} | {ic[0]:5.4f}±{ic[1]:2.2f} | {icir[0]:5.4f}±{icir[1]:2.2f}| {ric[0]:5.4f}±{ric[1]:2.2f} | {ricir[0]:5.4f}±{ricir[1]:2.2f} | {ar[0]:5.4f}±{ar[1]:2.2f} | {ir[0]:5.4f}±{ir[1]:2.2f}| {md[0]:5.4f}±{md[1]:2.2f} |\n"
    pprint(table)
    with open("table.md", "w") as f:
        f.write(table)
    return table


# read yaml, remove seed kwargs of model, and then save file in the temp_dir
def gen_yaml_file_without_seed_kwargs(yaml_path, temp_dir):
    with open(yaml_path, "r") as fp:
        config = yaml.safe_load(fp)
    try:
        del config["task"]["model"]["kwargs"]["seed"]
    except KeyError:
        # If the key does not exists, use original yaml
        # NOTE: it is very important if the model most run in original path(when sys.rel_path is used)
        return yaml_path
    else:
        # otherwise, generating a new yaml without random seed
        file_name = yaml_path.split("/")[-1]
        temp_path = os.path.join(temp_dir, file_name)
        with open(temp_path, "w") as fp:
            yaml.dump(config, fp)
        return temp_path


class ModelRunner:
    def _init_qlib(self, exp_folder_name):
        # init qlib
        GetData().qlib_data(exists_skip=True)
        qlib.init(
            exp_manager={
                "class": "MLflowExpManager",
                "module_path": "qlib.workflow.expm",
                "kwargs": {
                    "uri": "file:" + str(Path(os.getcwd()).resolve() / exp_folder_name),
                    "default_exp_name": "Experiment",
                },
            }
        )

    # function to run the all the models
    @only_allow_defined_args
    def run(
        self,
        times=1,
        models=None,
        dataset="Alpha360",
        exclude=False,
        qlib_uri: str = "git+https://github.com/microsoft/qlib#egg=pyqlib",
        exp_folder_name: str = "run_all_model_records",
        wait_before_rm_env: bool = False,
        wait_when_err: bool = False,
    ):
        """
        Please be aware that this function can only work under Linux. MacOS and Windows will be supported in the future.
        Any PR to enhance this method is highly welcomed. Besides, this script doesn't support parallel running the same model
        for multiple times, and this will be fixed in the future development.

        Parameters:
        -----------
        times : int
            determines how many times the model should be running.
        models : str or list
            determines the specific model or list of models to run or exclude.
        exclude : boolean
            determines whether the model being used is excluded or included.
        dataset : str
            determines the dataset to be used for each model.
        qlib_uri : str
            the uri to install qlib with pip
            it could be url on the we or local path (NOTE: the local path must be a absolute path)
        exp_folder_name: str
            the name of the experiment folder
        wait_before_rm_env : bool
            wait before remove environment.
        wait_when_err : bool
            wait when errors raised when executing commands

        Usage:
        -------
        Here are some use cases of the function in the bash:

        .. code-block:: bash

            # Case 1 - run all models multiple times
            python run_all_model.py run 3

            # Case 2 - run specific models multiple times
            python run_all_model.py run 3 mlp

            # Case 3 - run specific models multiple times with specific dataset
            python run_all_model.py run 3 mlp Alpha158

            # Case 4 - run other models except those are given as arguments for multiple times
            python run_all_model.py run 3 [mlp,tft,lstm] --exclude=True

            # Case 5 - run specific models for one time
            python run_all_model.py run --models=[mlp,lightgbm]

            # Case 6 - run other models except those are given as arguments for one time
            python run_all_model.py run --models=[mlp,tft,sfm] --exclude=True

        """
        self._init_qlib(exp_folder_name)

        # get all folders
        folders = get_all_folders(models, exclude)
        # init error messages:
        errors = dict()
        # run all the model for iterations
        for fn in folders:
            # get all files
            sys.stderr.write("Retrieving files...\n")
            yaml_path, req_path = get_all_files(folders[fn], dataset)
            if yaml_path is None:
                sys.stderr.write(f"There is no {dataset}.yaml file in {folders[fn]}")
                continue
            sys.stderr.write("\n")
            # create env by anaconda
            temp_dir, env_path, python_path, conda_activate = create_env()

            # install requirements.txt
            sys.stderr.write("Installing requirements.txt...\n")
            with open(req_path) as f:
                content = f.read()
            if "torch" in content:
                # automatically install pytorch according to nvidia's version
                execute(
                    f"{python_path} -m pip install light-the-torch", wait_when_err=wait_when_err
                )  # for automatically installing torch according to the nvidia driver
                execute(
                    f"{env_path / 'bin' / 'ltt'} install --install-cmd '{python_path} -m pip install {{packages}}' -- -r {req_path}",
                    wait_when_err=wait_when_err,
                )
            else:
                execute(f"{python_path} -m pip install -r {req_path}", wait_when_err=wait_when_err)
            sys.stderr.write("\n")

            # read yaml, remove seed kwargs of model, and then save file in the temp_dir
            yaml_path = gen_yaml_file_without_seed_kwargs(yaml_path, temp_dir)
            # setup gpu for tft
            if fn == "TFT":
                execute(
                    f"conda install -y --prefix {env_path} anaconda cudatoolkit=10.0 && conda install -y --prefix {env_path} cudnn",
                    wait_when_err=wait_when_err,
                )
                sys.stderr.write("\n")
            # install qlib
            sys.stderr.write("Installing qlib...\n")
            execute(f"{python_path} -m pip install --upgrade pip", wait_when_err=wait_when_err)  # TODO: FIX ME!
            execute(f"{python_path} -m pip install --upgrade cython", wait_when_err=wait_when_err)  # TODO: FIX ME!
            if fn == "TFT":
                execute(
                    f"cd {env_path} && {python_path} -m pip install --upgrade --force-reinstall --ignore-installed PyYAML -e {qlib_uri}",
                    wait_when_err=wait_when_err,
                )  # TODO: FIX ME!
            else:
                execute(
                    f"cd {env_path} && {python_path} -m pip install --upgrade --force-reinstall -e {qlib_uri}",
                    wait_when_err=wait_when_err,
                )  # TODO: FIX ME!
            sys.stderr.write("\n")
            # run workflow_by_config for multiple times
            for i in range(times):
                sys.stderr.write(f"Running the model: {fn} for iteration {i+1}...\n")
                errs = execute(
                    f"{python_path} {env_path / 'bin' / 'qrun'} {yaml_path} {fn} {exp_folder_name}",
                    wait_when_err=wait_when_err,
                )
                if errs is not None:
                    _errs = errors.get(fn, {})
                    _errs.update({i: errs})
                    errors[fn] = _errs
                sys.stderr.write("\n")
            # remove env
            sys.stderr.write(f"Deleting the environment: {env_path}...\n")
            if wait_before_rm_env:
                input("Press Enter to Continue")
            shutil.rmtree(env_path)
        # print errors
        sys.stderr.write(f"Here are some of the errors of the models...\n")
        pprint(errors)
        self._collect_results(exp_folder_name, dataset)

    def _collect_results(self, exp_folder_name, dataset):
        folders = get_all_folders(exp_folder_name, dataset)
        # getting all results
        sys.stderr.write(f"Retrieving results...\n")
        results = get_all_results(folders)
        if len(results) > 0:
            # calculating the mean and std
            sys.stderr.write(f"Calculating the mean and std of results...\n")
            results = cal_mean_std(results)
            # generating md table
            sys.stderr.write(f"Generating markdown table...\n")
            gen_and_save_md_table(results, dataset)
            sys.stderr.write("\n")
        sys.stderr.write("\n")
        # move results folder
        shutil.move(exp_folder_name, exp_folder_name + f"_{dataset}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
        shutil.move("table.md", f"table_{dataset}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.md")


if __name__ == "__main__":
    fire.Fire(ModelRunner)  # run all the model
