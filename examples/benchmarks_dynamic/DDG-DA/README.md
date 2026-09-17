# Introduction
This is the implementation of `DDG-DA` based on `Meta Controller` component provided by `Qlib`.

Please refer to the paper for more details: *DDG-DA: Data Distribution Generation for Predictable Concept Drift Adaptation* [[arXiv](https://arxiv.org/abs/2201.04038)]


# Background
In many real-world scenarios, we often deal with streaming data that is sequentially collected over time. Due to the non-stationary nature of the environment, the streaming data distribution may change in unpredictable ways, which is known as concept drift. To handle concept drift, previous methods first detect when/where the concept drift happens and then adapt models to fit the distribution of the latest data. However, there are still many cases that some underlying factors of environment evolution are predictable, making it possible to model the future concept drift trend of the streaming data, while such cases are not fully explored in previous work.

Therefore, we propose a novel method `DDG-DA`, that can effectively forecast the evolution of data distribution and improve the performance of models. Specifically, we first train a predictor to estimate the future data distribution, then leverage it to generate training samples, and finally train models on the generated data.

# Dataset
The data in the paper are private. So we conduct experiments on Qlib's public dataset.
Though the dataset is different, the conclusion remains the same. By applying `DDG-DA`, users can see rising trends at the test phase both in the proxy models' ICs and the performances of the forecasting models.

# Run the Code
After verifying that the recorder artifacts and local working files are your own
and cannot be replaced by untrusted writers (see below), run from this directory:
```bash
    python workflow.py --trusted_artifacts=True run
```

The default forecasting models are `Linear`. Users can choose other forecasting models by changing the `forecast_model` parameter when `DDG-DA` initializes. For example, users can try `LightGBM` forecasting models by running the following command:
```bash
    python workflow.py --trusted_artifacts=True --conf_path=../baseline/workflow_config_lightgbm_Alpha158.yaml run
```

## Recorder artifacts and local working files

`workflow.py` exposes the `DDGDA` workflow through the `DDGDABench` Fire entry
point. Its `trusted_artifacts` option defaults to `False`. Set it explicitly only
for artifacts and caches from a verified writer in access-controlled MLflow and
local storage: unrestricted pickle loading can execute code. Creating a run
yourself is not enough if someone else can overwrite its files.

The option covers recorder-backed executable meta-model/task loading, including
`InternalData.setup`, and DDG-DA's local handler/internal-data pickle cache reads;
prediction and label artifact loads remain restricted.
Lower-level callers can also pass `trusted_artifacts=True` to `MetaDatasetDS` or
`InternalData.setup`. A refused load is not a reason to retry automatically with
trust enabled. See the [recorder migration guide](https://qlib.readthedocs.io/en/latest/component/recorder.html#artifact-trust-migration)
for data compatibility and migration details.

This example also saves and reuses **local pickle files in `working_dir`**, which
the benchmark sets to this directory. Handler/internal-data caches default to
restricted loading, which refuses executable objects such as `Alpha158` or
`InternalData`. The explicit opt-in allows these caches to be restored with
ordinary pickle and emits a warning; it does not authenticate their contents.
Protect `working_dir`, the configuration directory (also used for handler
caching), and any supplied `h_path` from untrusted writes. Do not copy unknown
cached handlers, meta-information or models into them. There is no automatic
unsafe retry and no change to the global restricted loader. Other pickle APIs
and workflow YAML retain their own trust requirements; only use trusted
configurations and files.

Generated tasks keep lightweight handler-cache references, including the chosen
cache policy, rather than embedding the full market data. Treat saved task
configurations as executable inputs; reusing an opted-in task also reuses that
local-cache consent. Loading a saved task containing a reweighter through a
recorder still requires explicit recorder consent.

The Makefile's `clean` target deletes local pickle files and `mlruns`; preserve any
results you need before using it.

## Full workflow regression

From the repository root, with the test and model dependencies installed:

```bash
python -m pytest tests/rolling_tests/test_ddgda.py -m slow -q
```

The offline regression uses deterministic local daily market data and an isolated
MLflow store. Both linear and LightGBM similarity models run through feature
selection, seven similarity-training windows, daily rank IC, cache restoration,
30-epoch meta-training, inferred time weights, two rolling training windows,
prediction/label collection and a 40-day portfolio backtest. It checks default
refusal, explicit authorization, restored predictions, delayed replay of both
saved rolling tasks and non-empty numerical results without replacing workflow
stages with mocks. These small integration
cases verify functionality, not paper-scale performance or investment returns.

# Results
The results of related methods in Qlib's public dataset can be found [here](../)

# Requirements
Here are the minimal hardware requirements to run the ``workflow.py`` of DDG-DA.
* Memory: 45G
* Disk: 4G

Pytorch with CPU & RAM will be enough for this example.
