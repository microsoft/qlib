# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
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
    "numpy>=1.12.0, <1.24",
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
    # To ensure stable operation of the experiment manager, we have limited the version of mlflow,
    # and we need to verify whether version 2.0 of mlflow can serve qlib properly.
    "mlflow>=1.12.1, <=1.30.0",
    "tqdm",
    "loguru",
    "lightgbm>=3.3.0",
    "tornado",
    "joblib>=0.17.0",
    "ruamel.yaml>=0.16.12",
    "pymongo==3.7.2",  # For task management
    "scikit-learn>=0.22",
    "dill",
    "dataclasses;python_version<'3.7'",
    "filelock",
    "jinja2<3.1.0",  # for passing the readthedocs workflow.
    "gym",
    # Installing the latest version of protobuf for python versions below 3.8 will cause unit tests to fail.
    "protobuf<=3.20.1;python_version<='3.8'",
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
            "pre-commit",
            # CI dependencies
            "wheel",
            "setuptools",
            "black",
            "pylint",
            # Using the latest versions(0.981 and 0.982) of mypy,
            # the error "multiprocessing.Value()" is detected in the file "qlib/rl/utils/data_queue.py",
            # If this is fixed in a subsequent version of mypy, then we will revert to the latest version of mypy.
            # References: https://github.com/python/typeshed/issues/8799
            "mypy<0.981",
            "flake8",
            "nbqa",
            "jupyter",
            "nbconvert",
            # The 5.0.0 version of importlib-metadata removed the deprecated endpoint,
            # which prevented flake8 from working properly, so we restricted the version of importlib-metadata.
            # To help ensure the dependencies of flake8 https://github.com/python/importlib_metadata/issues/406
            "importlib-metadata<5.0.0",
            "readthedocs_sphinx_ext",
            "cmake",
            "lxml",
            "baostock",
            "yahooquery",
            "beautifulsoup4",
            # In version 0.4.11 of tianshou, the code:
            # logits, hidden = self.actor(batch.obs, state=state, info=batch.info)
            # was changed in PR787,
            # which causes pytest errors(AttributeError: 'dict' object has no attribute 'info') in CI,
            # so we restricted the version of tianshou.
            # References:
            # https://github.com/thu-ml/tianshou/releases
            "tianshou<=0.4.10",
            "gym>=0.24",  # If you do not put gym at the end, gym will degrade causing pytest results to fail.
        ],
        "rl": [
            "tianshou<=0.4.10",
            "torch",
        ],
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
