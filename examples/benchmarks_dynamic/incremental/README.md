# DoubleAdapt (KDD'23)
## Introduction
This is the official implementation of `DoubleAdapt`, an incremental learning framework for stock trend forecasting.

The paper has been accepted by KDD 2023. [[arXiv](https://arxiv.org/abs/2306.09862)]

## Dataset
Following DDG-DA, we run experiments on the crowd source version of qlib data which can be downloaded by
```bash
wget https://github.com/chenditc/investment_data/releases/download/2023-06-01/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/crowd_data --strip-components=2
```
Argument `--data_dir crowd_data` and `--data_dir cn_data` for crowd-source data and Yahoo-source data, respectively.

Argument `--alpha 360` or `--alpha 158` for Alpha360 and Alpha 158, respectively. 
> We have not carefully checked the program on Alpha158, possibly leaving bugs unresolved. 
> It is better to experiment on Alpha 360.
 
Note that we are to predict the stock trend **BUT NOT** the rank of stock trend, which is different from DDG-DA.

To this end, we use `CSZScoreNorm` in the learn_processors instead of `CSRankNorm`.

Pay attention to the argument `--rank_label False` (or `--rank_label True`) for the target label.

## Requirements

### Packages
```bash
conda install higher -c conda-forge
# pip install higher
```
Thanks to [github.com/facebookresearch/higher](https://github.com/facebookresearch/higher)

### RAM

This implementation requires ~8GB RAM on CSI500 when the update interval `step` is set to 20 trading days.

If your RAM is limited, you can split the function `dump_data` into two functions that dump training data and test data, respectively. 
Then, free the storage of training data before testing. 

Moveover, in our implementation, we cast all slices of stock data in `pandas.DataFrame` to `torch.Tensor` during data preprocessing.
This trick largely reduce CPU occupation during training and testing while it results in duplicate storage.

You can also set the argument `--preprocess_tensor False`, reducing RAM occupation to ~5GB (peak 8GB before training). 
Then, the data slices are created as virtual views of `pandas.DataFrame`, and the duplicates share the same memory address. 
Each batch will be casted as `torch.Tensor` when needed, requesting new memory of a tiny size.
However, `--preprocess_tensor False` can exhaust all cores of CPU and the speed is lower consequently.

### GPU Memory
DoubleAdapt requires at most 10GB GPU memory when `step` is set to 20. 
The occupation will be smaller on CSI300 and on the default Yahoo data (which bears more missing values).

If your GPU is limited, try to set a smaller `step` (e.g., 5) which may takes up ~2GB. And you can achieve higher performance.

> The reason why we set `step` to 20 rather than 5 is that 
RR and DDG-DA bear unaffordable time cost (e.g., 3 days for 10 runs) in experiments with `step` set to 5.   

## Remarks
### Carefully select `step` according to `horizon`
Argument `--horizon` decides the target label to be `Ref($close, -horizon-1}) / Ref($close, -1) - 1` in the China A-share market. 
Accordingly, there are always unknown ground-truth labels in the lasted `horizon` days of test data, and we can only use the rest for optimization of the meta-learners.
With a large `horizon` or a small `step`, the performance on the majority of the test data cannot be optimized, 
and the meta-learners may well be overfitted and shortsighted.
We provide an argument `--use_extra True` to take the nearest data as additional test data, while the improvement is often little.

It is recommended to let `step` be greater than `horizon` by at least 3 or 4, e.g., `--step 5 --horizon 1`.

> The current implementation **does not** support `step` to equal `horizon` (e.g., `--step 1 --horizon 1`).
> TODO: use $\phi^{k-2}$ and $\psi^{k-2}$ for the $k$-th online task

### Re-devise the data adapter for high-dimensional features
We experiment on Alpha360 where the feature adaptation involves 6$\times$6 affine transformation with a few parameters to learn.

In the case of high-dimensional features (e.g., Alpha158), the dense layers are too complex and bring only a little improvement. 
We recommend replacing the dense layers by [FiLM](https://arxiv.org/pdf/1709.07871.pdf) layers to reduce the complexity.

## Scripts
```bash
# Naive incremental learning
python -u main.py run_all --forecast_model GRU --market csi300 --data_dir crowd_data --rank_label False --naive True
# DoubleAdapt
python -u main.py run_all --forecast_model GRU --market csi300 --data_dir crowd_data --rank_label False -num_head 8 --tau 10
```
## Cite
If you find this useful for your work, please consider citing it as follows:
```bash
@InProceedings{DoubleAdapt,
  author       = {Lifan Zhao and Shuming Kong and Yanyan Shen},
  booktitle    = {Proceedings of the 29th {ACM} {SIGKDD} Conference on Knowledge Discovery and Data Mining},
  title        = {{DoubleAdapt}: A Meta-learning Approach to Incremental Learning for Stock Trend Forecasting},
  year         = {2023},
  month        = {aug},
  publisher    = {{ACM}},
  doi          = {10.1145/3580305.3599315},
}
```