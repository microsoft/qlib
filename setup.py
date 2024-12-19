from setuptools import find_packages, setup, Extension
import tomli
import numpy
import os


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

here = os.path.abspath(os.path.dirname(__file__))

with open("pyproject.toml", "rb") as f:
    pyproject_data = tomli.load(f)

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

project_config = pyproject_data.get("project", {})
name = project_config.get("name", "default-package-name")
description = project_config.get("description", "")
dependencies = project_config.get("dependencies", [])
classifiers = project_config.get("classifiers", [])
python_requires = project_config.get("requires-python", ">=3.8.0")
optional_dependencies = pyproject_data.get("project", {}).get("optional-dependencies", {})

NUMPY_INCLUDE = numpy.get_include()

VERSION = get_version("qlib/__init__.py")


setup(
    version=VERSION,
    name=name,
    description=description,
    install_requires=dependencies,
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    classifiers=classifiers,
    extras_require=optional_dependencies,
    python_requires=python_requires,
    license="MIT Licence",
    url="https://github.com/microsoft/qlib",
    packages=find_packages(exclude=("tests",)),
    entry_points={
        "console_scripts": [
            "qrun=qlib.workflow.cli:run",
        ],
    },
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
