# Portfolio Optimization Strategy

## Introduction

In `qlib/examples/benchmarks` we have various **alpha** models that predict
the stock returns. We also use a simple rule based `TopkDropoutStrategy` to
evaluate the investing performance of these models. However, such a strategy
is too simple to control the portfolio risk like correlation and volatility.

To this end, an optimization based strategy should be used to for the
trade-off between return and risk. In this doc, we will show how to use
`EnhancedIndexingStrategy` to maximize portfolio return while minimizing
tracking error relative to a benchmark.


## Preparation

We use China stock market data for our example.

1. Prepare CSI300 weight:

   ```bash
   wget https://github.com/SunsetWolf/qlib_dataset/releases/download/v0/csi300_weight.zip
   unzip -d ~/.qlib/qlib_data/cn_data csi300_weight.zip
   rm -f csi300_weight.zip
   ```
   NOTE:  We don't find any public free resource to get the weight in the benchmark. To run the example, we manually create this weight data.

2. Prepare risk model data:

   ```bash
   python prepare_riskdata.py
   ```

Here we use a **Statistical Risk Model** implemented in `qlib.model.riskmodel`.
However users are strongly recommended to use other risk models for better quality:
* **Fundamental Risk Model** like MSCI BARRA
* [Deep Risk Model](https://arxiv.org/abs/2107.05201)


## End-to-End Workflow

You can finish workflow with `EnhancedIndexingStrategy` by running
`qrun config_enhanced_indexing.yaml`.

In this config, we mainly changed the strategy section compared to
`qlib/examples/benchmarks/workflow_config_lightgbm_Alpha158.yaml`.
