import os

import numpy
from setuptools import Extension, setup
from setuptools_scm import get_version


def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), encoding="utf-8") as fp:
        return fp.read()


NUMPY_INCLUDE = numpy.get_include()


VERSION = get_version(root=".", relative_to=__file__)

# Define base requirements
install_requires = [
    "numpy>=1.12.0",
<<<<<<< HEAD
    "pandas>=0.25.1", 
=======
    "pandas>=0.25.1",
>>>>>>> f180e36a (fix: migrate from gym to gymnasium for NumPy 2.0+ compatibility)
    "scipy>=1.0.0",
    "scikit-learn>=0.22.0",
    "matplotlib>=3.0.0",
    "seaborn>=0.9.0",
    "tqdm",
    "joblib>=0.17.0",
    "ruamel.yaml>=0.16.0",
    "fire>=0.3.0",
    "cloudpickle",
    "lxml",
    "jinja2",
    "statsmodels",
    "plotly>=4.12.0",
    "redis>=3.0.1",
    "python-socketio",
    "pymongo>=3.7.0",
    "influxdb",
    "pyarrow>=6.0.0",
]

<<<<<<< HEAD
# Define RL-specific optional requirements  
=======
# Define RL-specific optional requirements
>>>>>>> f180e36a (fix: migrate from gym to gymnasium for NumPy 2.0+ compatibility)
extras_require = {
    "rl": [
        "gymnasium>=0.28.0",  # gymnasium
        "stable-baselines3>=1.2.0",
        "tensorboard>=2.0.0",
    ],
    "dev": [
        "black",
<<<<<<< HEAD
        "flake8", 
        "pytest",
        "pytest-cov",
        "sphinx",
    ]
=======
        "flake8",
        "pytest",
        "pytest-cov",
        "sphinx",
    ],
>>>>>>> f180e36a (fix: migrate from gym to gymnasium for NumPy 2.0+ compatibility)
}

setup(
    version=VERSION,
    ext_modules=[
        Extension(
            "qlib.data._libs.rolling",
            ["qlib/data/_libs/rolling.pyx"],
            language="c++",
            include_dirs=[NUMPY_INCLUDE],
        ),
        Extension(
            "qlib.data._libs.expanding",
            ["qlib/data/_libs/expanding.pyx"],
            language="c++",
            include_dirs=[NUMPY_INCLUDE],
        ),
    ],
)
