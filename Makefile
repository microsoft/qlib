# Prerequisite section
prerequisite:
	python -c "from setuptools import setup, Extension; from Cython.Build import cythonize; import numpy; extensions = [Extension('qlib.data._libs.rolling', ['qlib/data/_libs/rolling.pyx'], language='c++', include_dirs=[numpy.get_include()]), Extension('qlib.data._libs.expanding', ['qlib/data/_libs/expanding.pyx'], language='c++', include_dirs=[numpy.get_include()])]; setup(ext_modules=cythonize(extensions, language_level='3'), script_args=['build_ext', '--inplace'])"
