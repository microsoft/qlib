import os

import numpy
from setuptools import Extension, setup
from setuptools_scm import get_version


def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), encoding="utf-8") as fp:
        return fp.read()


NUMPY_INCLUDE = numpy.get_include()


VERSION = get_version(root=".", relativ_to=__file__).split("+")[0]

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
