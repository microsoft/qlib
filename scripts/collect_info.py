import sys
import platform
import qlib
import fire
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

QLIB_PATH = Path(__file__).absolute().resolve().parent.parent


class InfoCollector:
    """
    User could collect system info by following commands
    `cd scripts && python collect_info.py all`
    - NOTE: please avoid running this script in the project folder which contains `qlib`
    """

    def sys(self):
        """collect system related info"""
        for method in ["system", "machine", "platform", "version"]:
            print(getattr(platform, method)())

    def py(self):
        """collect Python related info"""
        print("Python version: {}".format(sys.version.replace("\n", " ")))

    def qlib(self):
        """collect qlib related info"""
        print("Qlib version: {}".format(qlib.__version__))
        REQUIRED = [
            "setuptools",
            "wheel",
            "cython",
            "pyyaml",
            "numpy",
            "pandas",
            "mlflow",
            "filelock",
            "redis",
            "dill",
            "fire",
            "ruamel.yaml",
            "python-redis-lock",
            "tqdm",
            "pymongo",
            "loguru",
            "lightgbm",
            "gym",
            "cvxpy",
            "joblib",
            "matplotlib",
            "jupyter",
            "nbconvert",
            "pyarrow",
            "pydantic-settings",
            "setuptools-scm",
        ]

        for package in REQUIRED:
            try:
                print(f"{package}=={version(package)}")
            except PackageNotFoundError:
                print(f"{package}: not installed")

    def all(self):
        """collect all info"""
        for method in ["sys", "py", "qlib"]:
            getattr(self, method)()
            print()


if __name__ == "__main__":
    fire.Fire(InfoCollector)
