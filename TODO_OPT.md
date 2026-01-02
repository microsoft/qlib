# Optimization TODO List

## 1. Dynamic Shuffling for MPS Compatibility
**Affected Models:** 
- `GeneralPTNN` (`qlib/contrib/model/pytorch_general_nn.py`)
- `ADARNN` (`qlib/contrib/model/pytorch_adarnn.py`)
- Any other models where `DataLoader(shuffle=True)` was disabled to fix MPS crashes.

**Issue:**
To resolve segmentation faults on macOS (MPS backend) caused by `DataLoader` shuffling non-contiguous memory, we disabled `shuffle=True` and implemented a one-time manual shuffle before the training loop. This means the training data order is fixed across all epochs, which is suboptimal compared to per-epoch shuffling.

**Optimization Goal:**
Restore the behavior of shuffling data at the beginning of **every epoch** to ensure better model convergence.

**Proposed Solution:**
- Modify the `fit` loop to re-shuffle indices and re-create the `DataLoader` (or use a custom `Sampler`) at the start of each epoch.
- Ensure that the shuffling mechanism respects the C-contiguous memory requirement for MPS.
