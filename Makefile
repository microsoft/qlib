.PHONY: prerequisite clean deepclean install init-qlib-env dev constraints black isort mypy ruff toml-sort lint pre-commit test-run test build upload docs-autobuild changelog docs-gen docs-mypy docs-coverage docs
#You can modify it according to your terminal
SHELL := /bin/bash

########################################################################################
# Variables
########################################################################################

# Documentation target directory, will be adapted to specific folder for readthedocs.
PUBLIC_DIR := $(shell [ "$$READTHEDOCS" = "True" ] && echo "$$READTHEDOCS_OUTPUT/html" || echo "public")

SO_DIR := qlib/data/_libs
SO_FILES := $(wildcard $(SO_DIR)/*.so)

########################################################################################
# Development Environment Management
########################################################################################
# Remove common intermediate files.
clean:
	-rm -rf \
		$(PUBLIC_DIR) \
		qlib/data/_libs \
		mlruns \
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
prerequisite:
	@if [ -n "$(SO_FILES)" ]; then \
		echo "Shared library files exist, skipping build."; \
	else \
		echo "No shared library files found, building..."; \
		python -m pip install cython numpy; \
		python -c "from setuptools import setup, Extension; from Cython.Build import cythonize; import numpy; extensions = [Extension('qlib.data._libs.rolling', ['qlib/data/_libs/rolling.pyx'], language='c++', include_dirs=[numpy.get_include()]), Extension('qlib.data._libs.expanding', ['qlib/data/_libs/expanding.pyx'], language='c++', include_dirs=[numpy.get_include()])]; setup(ext_modules=cythonize(extensions, language_level='3'), script_args=['build_ext', '--inplace'])"; \
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

install: prerequisite dependencies

dev: prerequisite lightgbm rl develop lint docs package

########################################################################################
# Lint and pre-commit
########################################################################################

# Check lint with black.
black:
	black . -l 120 --check --diff

# Check code folder with pylint.
pylint:
	pylint --disable=C0104,C0114,C0115,C0116,C0301,C0302,C0411,C0413,C1802,R0401,R0801,R0902,R0903,R0911,R0912,R0913,R0914,R0915,R0917,R1720,W0105,W0123,W0201,W0511,W0613,W1113,W1514,E0401,E1121,C0103,C0209,R0402,R1705,R1710,R1725,R1730,R1735,W0102,W0212,W0221,W0223,W0231,W0237,W0612,W0621,W0622,W0703,W1309,E1102,E1136 --const-rgx='[a-z_][a-z0-9_]{2,30}' qlib --init-hook="import astroid; astroid.context.InferenceContext.max_inferred = 500; import sys; sys.setrecursionlimit(2000)" || true
	pylint --disable=C0104,C0114,C0115,C0116,C0301,C0302,C0411,C0413,C1802,R0401,R0801,R0902,R0903,R0911,R0912,R0913,R0914,R0915,R0917,R1720,W0105,W0123,W0201,W0511,W0613,W1113,W1514,E0401,E1121,E1123,C0103,C0209,R0402,R1705,R1710,R1725,R1735,W0102,W0212,W0221,W0223,W0231,W0237,W0246,W0612,W0621,W0622,W0703,W1309,E1102,E1136 --const-rgx='[a-z_][a-z0-9_]{2,30}' scripts --init-hook="import astroid; astroid.context.InferenceContext.max_inferred = 500; import sys; sys.setrecursionlimit(2000)" || true

# Check code with flake8.
flake8:
	flake8 --ignore=E501,F541,E266,E402,W503,E731,E203 --per-file-ignores="__init__.py:F401,F403" qlib

# Check code with mypy.
mypy:
	mypy qlib --install-types --non-interactive || true
	mypy qlib --verbose || true

# Check ipynb with nbqa.
nbqa:
	nbqa black . -l 120 --check --diff
	nbqa pylint . --disable=C0104,C0114,C0115,C0116,C0301,C0302,C0411,C0413,C1802,R0401,R0801,R0902,R0903,R0911,R0912,R0913,R0914,R0915,R1720,W0105,W0123,W0201,W0511,W0613,W1113,W1514,E0401,E1121,C0103,C0209,R0402,R1705,R1710,R1725,R1735,W0102,W0212,W0221,W0223,W0231,W0237,W0612,W0621,W0622,W0703,W1309,E1102,E1136,W0719,W0104,W0404,C0412,W0611,C0410 --const-rgx='[a-z_][a-z0-9_]{2,30}$'

# Check ipynb with nbconvert.
nbconvert:
	jupyter nbconvert --to notebook --execute examples/workflow_by_code.ipynb

########################################################################################
# Package
########################################################################################

# Build the package.
build:
	python -m build

# Upload the package.
upload:
	python -m twine upload dist/*
