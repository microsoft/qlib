FROM continuumio/miniconda3:latest

WORKDIR /qlib

COPY . .

RUN apt-get update && \
    apt-get install -y build-essential

RUN conda create --name qlib_env python=3.8 -y
RUN echo "conda activate qlib_env" >> ~/.bashrc
ENV PATH /opt/conda/envs/qlib_env/bin:$PATH

RUN python -m pip install --upgrade pip

RUN python -m pip install numpy==1.23.5
RUN python -m pip install pandas==1.5.3
RUN python -m pip install importlib-metadata==5.2.0
RUN python -m pip install "cloudpickle<3"
RUN python -m pip install scikit-learn==1.3.2

RUN python -m pip install cython packaging tables matplotlib statsmodels
RUN python -m pip install pybind11 cvxpy

ARG IS_STABLE="yes"

RUN if [ "$IS_STABLE" = "yes" ]; then \
        python -m pip install pyqlib; \
    else \
        python setup.py install; \
    fi
