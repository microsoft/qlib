# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import io
import os
import numpy

from setuptools import find_packages, setup, Extension


def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), encoding="utf-8") as fp:
        return fp.read()


def get_version(rel_path: str) -> str:
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


# Package meta-data.
NAME = "pyqlib"
DESCRIPTION = "A Quantitative-research Platform"
REQUIRES_PYTHON = ">=3.5.0"

from pathlib import Path
from shutil import copyfile

VERSION = get_version("qlib/__init__.py")

# Detect Cython
try:
    import Cython

    ver = Cython.__version__
    _CYTHON_INSTALLED = ver >= "0.28"
except ImportError:
    _CYTHON_INSTALLED = False

if not _CYTHON_INSTALLED:
    print("Required Cython version >= 0.28 is not detected!")
    print('Please run "pip install --upgrade cython" first.')
    exit(-1)

# What packages are required for this module to be executed?
# `estimator` may depend on other packages. In order to reduce dependencies, it is not written here.
REQUIRED = [
    "numpy>=1.12.0",
    "pandas>=0.25.1",
    "scipy>=1.0.0",
    "requests>=2.18.0",
    "sacred>=0.7.4",
    "python-socketio",
    "redis>=3.0.1",
    "python-redis-lock>=3.3.1",
    "schedule>=0.6.0",
    "cvxpy>=1.0.21",
    "hyperopt==0.1.2",
    "fire>=0.3.1",
    "statsmodels",
    "xlrd>=1.0.0",
    "plotly>=4.12.0",
    "matplotlib>=3.3",
    "tables>=3.6.1",
    "pyyaml>=5.3.1",
    "mlflow>=1.12.1",
    "tqdm",
    "loguru",
    "lightgbm",
    "tornado",
    "joblib>=0.17.0",
    "ruamel.yaml>=0.16.12",
    "pymongo==3.7.2",  # For task management
    "scikit-learn>=0.22",
    "dill",
    "dataclasses;python_version<'3.7'",
    "filelock",
]

# Numpy include
NUMPY_INCLUDE = numpy.get_include()

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


# Cython Extensions
extensions = [
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
]

# Where the magic happens:
setup(
    name=NAME,
    version=VERSION,
    license="MIT Licence",
    url="https://github.com/microsoft/qlib",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(exclude=("tests",)),
    # if your package is a single module, use this instead of 'packages':
    # py_modules=['qlib'],
    entry_points={
        # 'console_scripts': ['mycli=mymodule:cli'],
        "console_scripts": [
            "qrun=qlib.workflow.cli:run",
        ],
    },
    ext_modules=extensions,
    install_requires=REQUIRED,
    extras_require={
        "dev": [
            "coverage",
            "pytest>=3",
            "sphinx",
            "sphinx_rtd_theme",
        ]
    },
    include_package_data=True,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        # 'License :: OSI Approved :: MIT License',
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
