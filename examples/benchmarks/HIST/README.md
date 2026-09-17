# HIST
* Code: [https://github.com/Wentao-Xu/HIST](https://github.com/Wentao-Xu/HIST)
* Paper: [HIST: A Graph-based Framework for Stock Trend Forecasting via Mining Concept-Oriented Shared InformationAdaRNN: Adaptive Learning and Forecasting for Time Series](https://arxiv.org/abs/2110.13716).

## Stock-index mapping migration

The bundled `qlib_csi300_stock_index.npy` object mapping has been replaced by
[`qlib_csi300_stock_index.json`](qlib_csi300_stock_index.json), preserving all
**735 entries**. The [example YAML](workflow_config_hist_Alpha360.yaml) already
uses the new file. Update the same setting in your own workflow YAML:

```yaml
task:
    model:
        kwargs:
            stock_index: "benchmarks/HIST/qlib_csi300_stock_index.json"
```

This path follows the example's convention of running from `examples/`; adjust it
to your working directory. `stock2concept` is a separate numeric matrix and still
uses `.npy`. Do not rename or replace that matrix with the JSON mapping.

For a **known-trusted custom mapping**, re-export it from your original trusted
metadata or producer into a JSON object:

- Keys must be instrument strings, matching your Qlib dataset.
- Values must be non-negative integers (not strings, floats or booleans) indexing
  rows in the corresponding `stock2concept` matrix.
- Preserve each instrument's original row assignment, and verify all indices are
  within the matrix's row bounds.
- Point `task.model.kwargs.stock_index` at your new `.json` file and keep the
  matching concept matrix.

If you only have an old object-pickled `.npy`, recover or regenerate the mapping
from the trusted source rather than loading an unknown file to convert it.
Changing the filename extension alone does not convert the contents. HIST
deliberately rejects the old object format because it requires executable pickle
deserialization; there is no fallback or recorder trust flag that re-enables it.

See the [recorder migration guide](https://qlib.readthedocs.io/en/latest/component/recorder.html#artifact-trust-migration)
for the separate policy on model/dataset artifacts. The JSON mapping change does
not make pre-existing local model checkpoints or other pickle loaders safe; those
inputs still require independent trust.

## Full workflow regression

From the repository root, with the test and model dependencies installed:

```bash
python -m pytest tests/model/test_hist_workflow.py -m slow -q
```

This offline CPU regression runs real Alpha360/DatasetH preparation, one HIST
training epoch, signal analysis and a six-day TopkDropout backtest in an isolated
MLflow store. It checks 48 prediction/label rows, actual optimizer updates,
finite reports and trading activity. Saved model/dataset objects are refused by
default; explicitly trusted reloads reproduce predictions exactly, including in
a fresh Python process. Metadata and concept fixtures are local JSON and numeric
NumPy files, with no downloads. This is functional integration coverage, not a
paper-scale model-quality benchmark.