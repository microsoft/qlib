#!/bin/sh
git clone https://github.com/microsoft/qlib.git
cd qlib
ls
pip install pyqlib
# or
# pip install numpy
# pip install --upgrade cython
# python setup.py install
cd examples
ls
qrun benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml