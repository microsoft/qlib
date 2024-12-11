# Prerequisite section
prerequisite:
	pip install numpy pytest
	pip install --upgrade cython
	ln -s `python -c "import numpy; print(numpy.get_include())"` numpy_include
