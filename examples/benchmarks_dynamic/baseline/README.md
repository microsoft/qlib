# Introduction

This is the framework of periodically Rolling Retrain (RR) forecasting models. RR adapts to market dynamics by utilizing the up-to-date data periodically.

## Run the Code
Users can try RR by running the following command:
```bash
    python rolling_benchmark.py run
```

The default forecasting models are `Linear`. Users can choose other forecasting models by changing the `model_type` parameter.
For example, users can try `LightGBM` forecasting models by running the following command:
```bash
    python rolling_benchmark.py --conf_path=workflow_config_lightgbm_Alpha158.yaml run

```
