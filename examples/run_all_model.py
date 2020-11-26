#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import os
import sys
import fire
import venv
import glob
import shutil
import tempfile
import statistics
from pathlib import Path
from operator import xor
from subprocess import Popen, PIPE
from threading import Thread
from pprint import pprint
from urllib.parse import urlparse
from urllib.request import urlretrieve

import qlib
from qlib.config import REG_CN
from qlib.workflow import R
from qlib.workflow.cli import workflow
from qlib.utils import exists_qlib_data

# init qlib
provider_uri = "~/.qlib/qlib_data/cn_data"
if not exists_qlib_data(provider_uri):
    print(f"Qlib data is not found in {provider_uri}")
    sys.path.append(str(Path(__file__).resolve().parent.parent.joinpath("scripts")))
    from get_data import GetData

    GetData().qlib_data(target_dir=provider_uri, region=REG_CN)
qlib.init(provider_uri=provider_uri, region=REG_CN)


class ExtendedEnvBuilder(venv.EnvBuilder):
    """
    Thie class is modified based on https://docs.python.org/3/library/venv.html.
    This builder installs setuptools and pip so that you can pip or
    easy_install other packages into the created virtual environment.

    :param nodist: If true, setuptools and pip are not installed into the
                   created virtual environment.
    :param nopip: If true, pip is not installed into the created
                  virtual environment.
    :param progress: If setuptools or pip are installed, the progress of the
                     installation can be monitored by passing a progress
                     callable. If specified, it is called with two
                     arguments: a string indicating some progress, and a
                     context indicating where the string is coming from.
                     The context argument can have one of three values:
                     'main', indicating that it is called from virtualize()
                     itself, and 'stdout' and 'stderr', which are obtained
                     by reading lines from the output streams of a subprocess
                     which is used to install the app.

                     If a callable is not specified, default progress
                     information is output to sys.stderr.
    """

    def __init__(self, *args, **kwargs):
        self.nodist = kwargs.pop("nodist", False)
        self.nopip = kwargs.pop("nopip", False)
        self.progress = kwargs.pop("progress", None)
        self.verbose = kwargs.pop("verbose", False)
        super().__init__(*args, **kwargs)

    def post_setup(self, context):
        """
        Set up any packages which need to be pre-installed into the
        virtual environment being created.

        :param context: The information for the virtual environment
                        creation request being processed.
        """
        os.environ["VIRTUAL_ENV"] = context.env_dir
        if not self.nodist:
            self.install_setuptools(context)
        # Can't install pip without setuptools
        if not self.nopip and not self.nodist:
            self.install_pip(context)

    def reader(self, stream, context):
        """
        Read lines from a subprocess' output stream and either pass to a progress
        callable (if specified) or write progress information to sys.stderr.
        """
        progress = self.progress
        while True:
            s = stream.readline()
            if not s:
                break
            if progress is not None:
                progress(s, context)
            else:
                if not self.verbose:
                    sys.stderr.write(".")
                else:
                    sys.stderr.write(s.decode("utf-8"))
                sys.stderr.flush()
        stream.close()

    def install_script(self, context, name, url):
        _, _, path, _, _, _ = urlparse(url)
        fn = os.path.split(path)[-1]
        binpath = context.bin_path
        distpath = os.path.join(binpath, fn)
        # Download script into the virtual environment's binaries folder
        urlretrieve(url, distpath)
        progress = self.progress
        if self.verbose:
            term = "\n"
        else:
            term = ""
        if progress is not None:
            progress("Installing %s ...%s" % (name, term), "main")
        else:
            sys.stderr.write("Installing %s ...%s" % (name, term))
            sys.stderr.flush()
        # Install in the virtual environment
        args = [context.env_exe, fn]
        p = Popen(args, stdout=PIPE, stderr=PIPE, cwd=binpath)
        t1 = Thread(target=self.reader, args=(p.stdout, "stdout"))
        t1.start()
        t2 = Thread(target=self.reader, args=(p.stderr, "stderr"))
        t2.start()
        p.wait()
        t1.join()
        t2.join()
        if progress is not None:
            progress("done.", "main")
        else:
            sys.stderr.write("done.\n")
        # Clean up - no longer needed
        os.unlink(distpath)

    def install_setuptools(self, context):
        """
        Install setuptools in the virtual environment.

        :param context: The information for the virtual environment
                        creation request being processed.
        """
        url = "https://bootstrap.pypa.io/ez_setup.py"
        self.install_script(context, "setuptools", url)
        # clear up the setuptools archive which gets downloaded
        pred = lambda o: o.startswith("setuptools-") and o.endswith(".tar.gz")
        files = filter(pred, os.listdir(context.bin_path))
        for f in files:
            f = os.path.join(context.bin_path, f)
            os.unlink(f)

    def install_pip(self, context):
        """
        Install pip in the virtual environment.

        :param context: The information for the virtual environment
                        creation request being processed.
        """
        url = "https://bootstrap.pypa.io/get-pip.py"
        self.install_script(context, "pip", url)


# function to check cuda version on the machine, this case is for the model TFT
def check_cuda(folders):
    path = "/usr/local/cuda/version.txt"
    exclude_tft = True
    if os.path.exists(path):
        with open(path, "w") as f:
            if "10.1" in str(f.read()) or "10.0" in str(f.read()):
                exclude_tft = False
    if exclude_tft and "TFT" in folders:
        del folders["TFT"]
    return folders


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
    folders = check_cuda(folders)
    return folders


# function to get all the files under the model folder
def get_all_files(folder_path) -> (str, str):
    yaml_path = str(Path(f"{folder_path}") / "*.yaml")
    req_path = str(Path(f"{folder_path}") / "*.txt")
    return glob.glob(yaml_path)[0], glob.glob(req_path)[0]


# function to retrieve all the results
def get_all_results(folders) -> dict:
    results = dict()
    for fn in folders:
        exp = R.get_exp(experiment_name=fn, create=False)
        recorders = exp.list_recorders()
        result = dict()
        result["annualized_return_with_cost"] = list()
        result["information_ratio_with_cost"] = list()
        result["max_drawdown_with_cost"] = list()
        for recorder_id in recorders:
            recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=fn)
            metrics = recorder.list_metrics()
            result["annualized_return_with_cost"].append(metrics["excess_return_with_cost.annualized_return"])
            result["information_ratio_with_cost"].append(metrics["excess_return_with_cost.information_ratio"])
            result["max_drawdown_with_cost"].append(metrics["excess_return_with_cost.max_drawdown"])
        results[fn] = result
    return results


# function to generate and save markdown table
def gen_and_save_md_table(metrics):
    table = "| Model Name | Annualized Return | Information Ratio | Max Drawdown |\n"
    table += "|---|---|---|---|\n"
    for fn in metrics:
        ar = metrics[fn]["annualized_return_with_cost"]
        ir = metrics[fn]["information_ratio_with_cost"]
        md = metrics[fn]["max_drawdown_with_cost"]
        table += f"| {fn} | {ar[0]:9.4f}±{ar[1]:9.2f} | {ir[0]:9.4f}±{ir[1]:9.2f}| {md[0]:9.4f}±{md[1]:9.2f} |\n"
    pprint(table)
    with open("table.md", "w") as f:
        f.write(table)
    return table


# function to run the all the models
def run(times=1, models=None, exclude=False):
    """
    Please be aware that this function can only work under Linux. MacOS and Windows will be supported in the future.
    Any PR to enhance this method is highly welcomed.

    Parameters:
    -----------
    times : int
        determines how many times the model should be running.
    models : str or list
        determines the specific model or list of models to run or exclude.
    exclude : boolean
        determines whether the model being used is excluded or included.

    Usage:
    -------
    Here are some use cases of the function in the bash:

    .. code-block:: bash

        # Case 1 - run all models multiple times
        python run_all_model.py 3

        # Case 2 - run specific models multiple times
        python run_all_model.py 3 dnn

        # Case 3 - run other models except those are given as arguments for multiple times
        python run_all_model.py 3 [dnn,tft,lstm] True

        # Case 4 - run specific models for one time
        python run_all_model.py --models=[dnn,lightgbm]

        # Case 5 - run other models except those are given as aruments for one time
        python run_all_model.py --models=[dnn,tft,sfm] --exclude=True

    """
    # get all folders
    folders = get_all_folders(models, exclude)
    # set up
    compatible = True
    if sys.version_info < (3, 3):
        compatible = False
    elif not hasattr(sys, "base_prefix"):
        compatible = False
    if not compatible:
        raise ValueError("This script is only for use with " "Python 3.3 or later")
    if os.name == "nt":
        use_symlinks = False
    else:
        use_symlinks = True
    builder = ExtendedEnvBuilder(
        system_site_packages=False,
        clear=False,
        symlinks=use_symlinks,
        upgrade=False,
        nodist=False,
        nopip=False,
        verbose=False,
    )
    # run all the model for iterations
    for fn in folders:
        # create env
        temp_dir = tempfile.mkdtemp()
        env_path = Path(temp_dir).absolute()
        sys.stderr.write(f"Creating Virtual Environment with path: {env_path}...\n")
        builder.create(str(env_path))
        python_path = env_path / "bin" / "python"  # TODO: FIX ME!
        sys.stderr.write("\n")
        # get all files
        sys.stderr.write("Retrieving files...\n")
        yaml_path, req_path = get_all_files(folders[fn])
        sys.stderr.write("\n")
        # install requirements.txt
        sys.stderr.write("Installing requirements.txt...\n")
        os.system(f"{python_path} -m pip install -r {req_path}")
        sys.stderr.write("\n")
        # install qlib
        sys.stderr.write("Installing qlib...\n")
        os.system(f"{python_path} -m pip install --upgrade cython")  # TODO: FIX ME!
        os.system(f"{python_path} -m pip install -e git+https://github.com/you-n-g/qlib#egg=pyqlib")  # TODO: FIX ME!
        sys.stderr.write("\n")
        # run workflow_by_config for multiple times
        for i in range(times):
            sys.stderr.write(f"Running the model: {fn} for iteration {i+1}...\n")
            os.system(f"{python_path} {env_path / 'src/pyqlib/qlib/workflow/cli.py'} {yaml_path} {fn}")
            sys.stderr.write("\n")
        # remove env
        sys.stderr.write(f"Deleting the environment: {env_path}...\n")
        shutil.rmtree(env_path)
    # getting all results
    sys.stderr.write(f"Retrieving results...\n")
    results = get_all_results(folders)
    # calculating the mean and std
    sys.stderr.write(f"Calculating the mean and std of results...\n")
    results = cal_mean_std(results)
    # generating md table
    sys.stderr.write(f"Generating markdown table...\n")
    gen_and_save_md_table(results)


if __name__ == "__main__":
    rc = 1
    try:
        fire.Fire(run)  # run all the model
        rc = 0
    except Exception as e:
        print("Error: %s" % e, file=sys.stderr)
    sys.exit(rc)
