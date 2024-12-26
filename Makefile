.PHONY: clean deepclean prerequisite dependencies lightgbm rl develop lint docs package test analysis all install dev black pylint flake8 mypy nbqa nbconvert lint build upload docs-gen
#You can modify it according to your terminal
SHELL := /bin/bash

########################################################################################
# Variables
########################################################################################

# Documentation target directory, will be adapted to specific folder for readthedocs.
PUBLIC_DIR := $(shell [ "$$READTHEDOCS" = "True" ] && echo "$$READTHEDOCS_OUTPUT/html" || echo "public")

SO_DIR := qlib/data/_libs
SO_FILES := $(wildcard $(SO_DIR)/*.so)

ifeq ($(OS),Windows_NT)
    IS_WINDOWS = true
else
    IS_WINDOWS = false
endif

########################################################################################
# Development Environment Management
########################################################################################
# Remove common intermediate files.
clean:
	-rm -rf \
		$(PUBLIC_DIR) \
		qlib/data/_libs/*.cpp \
		qlib/data/_libs/*.so \
		mlruns \
		public \
		build \
		.coverage \
		.mypy_cache \
		.pytest_cache \
		.ruff_cache \
		Pipfile* \
		coverage.xml \
		dist \
		release-notes.md

	find . -name '*.egg-info' -print0 | xargs -0 rm -rf
	find . -name '*.pyc' -print0 | xargs -0 rm -f
	find . -name '*.swp' -print0 | xargs -0 rm -f
	find . -name '.DS_Store' -print0 | xargs -0 rm -f
	find . -name '__pycache__' -print0 | xargs -0 rm -rf

# Remove pre-commit hook, virtual environment alongside itermediate files.
deepclean: clean
	if command -v pre-commit > /dev/null 2>&1; then pre-commit uninstall --hook-type pre-push; fi
	if command -v pipenv >/dev/null 2>&1 && pipenv --venv >/dev/null 2>&1; then pipenv --rm; fi

# Prerequisite section
# What this code does is compile two Cython modules, rolling and expanding, using setuptools and Cython,
# and builds them as binary expansion modules that can be imported directly into Python.
# Since pyproject.toml can't do that, we compile it here.

# pywinpty as a dependency of jupyter on windows, if you use pip install pywinpty installation,
# will first download the tar.gz file, and then locally compiled and installed,
# this will lead to some unnecessary trouble, so we choose to install the compiled whl file, to avoid trouble.
prerequisite:
	@if [ -n "$(SO_FILES)" ]; then \
		echo "Shared library files exist, skipping build."; \
	else \
		echo "No shared library files found, building..."; \
		pip install --upgrade setuptools wheel; \
		python -m pip install cython numpy; \
		python -c "from setuptools import setup, Extension; from Cython.Build import cythonize; import numpy; extensions = [Extension('qlib.data._libs.rolling', ['qlib/data/_libs/rolling.pyx'], language='c++', include_dirs=[numpy.get_include()]), Extension('qlib.data._libs.expanding', ['qlib/data/_libs/expanding.pyx'], language='c++', include_dirs=[numpy.get_include()])]; setup(ext_modules=cythonize(extensions, language_level='3'), script_args=['build_ext', '--inplace'])"; \
	fi

	@if [ "$(IS_WINDOWS)" = "true" ]; then \
		python -m pip install pywinpty --only-binary=:all:; \
	fi

# Install the package in editable mode.
dependencies:
	python -m pip install -e .

lightgbm:
	python -m pip install lightgbm --prefer-binary

rl:
	python -m pip install -e .[rl]

develop:
	python -m pip install -e .[dev]

lint:
	python -m pip install -e .[lint]

docs:
	python -m pip install -e .[docs]

package:
	python -m pip install -e .[package]

test:
	python -m pip install -e .[test]

analysis:
	python -m pip install -e .[analysis]

all:
	python -m pip install -e .[pywinpty,dev,lint,docs,package,test,analysis,rl]

install: prerequisite dependencies

dev: prerequisite all

########################################################################################
# Lint and pre-commit
########################################################################################

# Check lint with black.
black:
	black . -l 120 --check --diff

# Check code folder with pylint.
# TODO: These problems we will solve in the future. Important among them are: W0221, W0223, W0237, E1102
# 	C0103: invalid-name
# 	C0209: consider-using-f-string
# 	R0402: consider-using-from-import
# 	R1705: no-else-return
# 	R1710: inconsistent-return-statements
# 	R1725: super-with-arguments
# 	R1735: use-dict-literal
# 	W0102: dangerous-default-value
# 	W0212: protected-access
# 	W0221: arguments-differ
# 	W0223: abstract-method
# 	W0231: super-init-not-called
# 	W0237: arguments-renamed
# 	W0612: unused-variable
# 	W0621: redefined-outer-name
# 	W0622: redefined-builtin
# 	FIXME: specify exception type
# 	W0703: broad-except
# 	W1309: f-string-without-interpolation
# 	E1102: not-callable
# 	E1136: unsubscriptable-object
# 	W4904: deprecated-class
# 	R0917: too-many-positional-arguments
# 	E1123: unexpected-keyword-arg
# References for disable error: https://pylint.pycqa.org/en/latest/user_guide/messages/messages_overview.html
# We use sys.setrecursionlimit(2000) to make the recursion depth larger to ensure that pylint works properly (the default recursion depth is 1000).
# References for parameters: https://github.com/PyCQA/pylint/issues/4577#issuecomment-1000245962
pylint:
	pylint --disable=C0104,C0114,C0115,C0116,C0301,C0302,C0411,C0413,C1802,R0401,R0801,R0902,R0903,R0911,R0912,R0913,R0914,R0915,R0917,R1720,W0105,W0123,W0201,W0511,W0613,W1113,W1514,W4904,E0401,E1121,C0103,C0209,R0402,R1705,R1710,R1725,R1730,R1735,W0102,W0212,W0221,W0223,W0231,W0237,W0612,W0621,W0622,W0703,W1309,E1102,E1136 --const-rgx='[a-z_][a-z0-9_]{2,30}' qlib --init-hook="import astroid; astroid.context.InferenceContext.max_inferred = 500; import sys; sys.setrecursionlimit(2000)"
	pylint --disable=C0104,C0114,C0115,C0116,C0301,C0302,C0411,C0413,C1802,R0401,R0801,R0902,R0903,R0911,R0912,R0913,R0914,R0915,R0917,R1720,W0105,W0123,W0201,W0511,W0613,W1113,W1514,E0401,E1121,E1123,C0103,C0209,R0402,R1705,R1710,R1725,R1735,W0102,W0212,W0221,W0223,W0231,W0237,W0246,W0612,W0621,W0622,W0703,W1309,E1102,E1136 --const-rgx='[a-z_][a-z0-9_]{2,30}' scripts --init-hook="import astroid; astroid.context.InferenceContext.max_inferred = 500; import sys; sys.setrecursionlimit(2000)"

# Check code with flake8.
# The following flake8 error codes were ignored:
# E501 line too long
# 	Description: We have used black to limit the length of each line to 120.
# F541 f-string is missing placeholders
# 	Description: The same thing is done when using pylint for detection.
# E266 too many leading '#' for block comment
# 	Description: To make the code more readable, a lot of "#" is used.
#         This error code appears centrally in:
# 			qlib/backtest/executor.py
# 			qlib/data/ops.py
# 			qlib/utils/__init__.py
# E402 module level import not at top of file
# 	Description: There are times when module level import is not available at the top of the file.
# W503 line break before binary operator
# 	Description: Since black formats the length of each line of code, it has to perform a line break when a line of arithmetic is too long.
# E731 do not assign a lambda expression, use a def
# 	Description: Restricts the use of lambda expressions, but at some point lambda expressions are required.
# E203 whitespace before ':'
# 	Description: If there is whitespace before ":", it cannot pass the black check.
flake8:
	flake8 --ignore=E501,F541,E266,E402,W503,E731,E203 --per-file-ignores="__init__.py:F401,F403" qlib

# Check code with mypy.
# https://github.com/python/mypy/issues/10600
mypy:
	mypy qlib --install-types --non-interactive
	mypy qlib --verbose

# Check ipynb with nbqa.
nbqa:
	nbqa black . -l 120 --check --diff
	nbqa pylint . --disable=C0104,C0114,C0115,C0116,C0301,C0302,C0411,C0413,C1802,R0401,R0801,R0902,R0903,R0911,R0912,R0913,R0914,R0915,R1720,W0105,W0123,W0201,W0511,W0613,W1113,W1514,E0401,E1121,C0103,C0209,R0402,R1705,R1710,R1725,R1735,W0102,W0212,W0221,W0223,W0231,W0237,W0612,W0621,W0622,W0703,W1309,E1102,E1136,W0719,W0104,W0404,C0412,W0611,C0410 --const-rgx='[a-z_][a-z0-9_]{2,30}'

# Check ipynb with nbconvert.(Run after data downloads)
# TODO: Add more ipynb files in future
nbconvert:
	jupyter nbconvert --to notebook --execute examples/workflow_by_code.ipynb

lint: black pylint flake8 mypy nbqa

########################################################################################
# Package
########################################################################################

# Build the package.
build:
	python -m build --wheel

# Upload the package.
upload:
	python -m twine upload dist/*

########################################################################################
# Documentation
########################################################################################

docs-gen:
	python -m sphinx.cmd.build -W docs $(PUBLIC_DIR)
