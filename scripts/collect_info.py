import sys
import platform
import qlib
import fire
import pkg_resources
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
            "numpy",
            "pandas",
            "scipy",
            "requests",
            "sacred",
            "python-socketio",
            "redis",
            "python-redis-lock",
            "schedule",
            "cvxpy",
            "hyperopt",
            "fire",
            "statsmodels",
            "xlrd",
            "plotly",
            "matplotlib",
            "tables",
            "pyyaml",
            "mlflow",
            "tqdm",
            "loguru",
            "lightgbm",
            "tornado",
            "joblib",
            "fire",
            "ruamel.yaml",
        ]

        for package in REQUIRED:
            version = pkg_resources.get_distribution(package).version
            print(f"{package}=={version}")

    def all(self):
        """collect all info"""
        for method in ["sys", "py", "qlib"]:
            getattr(self, method)()
            print()


if __name__ == "__main__":
    fire.Fire(InfoCollector)
