# Temporal Fusion Transformers Benchmark
## Source
**Reference**: Lim, Bryan, et al. "Temporal fusion transformers for interpretable multi-horizon time series forecasting." arXiv preprint arXiv:1912.09363 (2019).

**GitHub**: https://github.com/google-research/google-research/tree/master/tft

## Run the Workflow
Users can follow the ``workflow_by_code_tft.py`` to run the benchmark. 

### Notes
1. Please be **aware** that this script can only support `Python 3.5 - 3.8`.
2. If the CUDA version on your machine is not 10.0, please remember to run the following commands `conda install anaconda cudatoolkit=10.0` and `conda install cudnn` on your machine.
3. The model must run in GPU, or an error will be raised.
4. New datasets should be registered in ``data_formatters``, for detail please visit the source.
