# Benchmarks Performance

Here are the results of each benchmark model running on Qlib's `Alpha360` and `Alpha158` dataset with China's A shared-stock & CSI300 data respectively. The values of each metric are the mean and std calculated based on 20 runs with different random seeds.

The numbers shown below demonstrate the performance of the entire `workflow` of each model. We will update the `workflow` as well as models in the near future for better results.

> If you need to reproduce the results below, please use the **v1** dataset: `python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/qlib_cn_1d --region cn --version v1`
>
> In the new version of qlib, the default dataset is **v2**. Since the data is collected from the YahooFinance API (which is not very stable), the results of *v2* and *v1* may differ

## Alpha360 dataset
| Model Name | Dataset | IC | ICIR | Rank IC | Rank ICIR | Annualized Return | Information Ratio | Max Drawdown |
|---|---|---|---|---|---|---|---|---|
| Linear | Alpha360 | 0.0150±0.00 | 0.1049±0.00| 0.0284±0.00 | 0.1970±0.00 | -0.0659±0.00 | -0.7072±0.00| -0.2955±0.00 |
| CatBoost (Liudmila Prokhorenkova, et al.) | Alpha360 | 0.0397±0.00 | 0.2878±0.00| 0.0470±0.00 | 0.3703±0.00 | 0.0342±0.00 | 0.4092±0.00| -0.1057±0.00 |
| XGBoost (Tianqi Chen, et al.) | Alpha360 | 0.0400±0.00 | 0.3031±0.00| 0.0461±0.00 | 0.3862±0.00 | 0.0528±0.00 | 0.6307±0.00| -0.1113±0.00 |
| LightGBM (Guolin Ke, et al.) | Alpha360 | 0.0399±0.00 | 0.3075±0.00| 0.0492±0.00 | 0.4019±0.00 | 0.0323±0.00 | 0.4370±0.00| -0.0917±0.00 |
| MLP | Alpha360 | 0.0285±0.00 | 0.1981±0.02| 0.0402±0.00 | 0.2993±0.02 | 0.0073±0.02 | 0.0880±0.22| -0.1446±0.03 |
| GRU (Kyunghyun Cho, et al.) | Alpha360 | 0.0490±0.01 | 0.3787±0.05| 0.0581±0.00 | 0.4664±0.04 | 0.0726±0.02 | 0.9817±0.34| -0.0902±0.03 |
| LSTM (Sepp Hochreiter, et al.) | Alpha360 | 0.0443±0.01 | 0.3401±0.05| 0.0536±0.01 | 0.4248±0.05 | 0.0627±0.03 | 0.8441±0.48| -0.0882±0.03 |
| ALSTM (Yao Qin, et al.) | Alpha360 | 0.0493±0.01 | 0.3778±0.06| 0.0585±0.00 | 0.4606±0.04 | 0.0513±0.03 | 0.6727±0.38| -0.1085±0.02 |
| GATs (Petar Velickovic, et al.) | Alpha360 | 0.0475±0.00 | 0.3515±0.02| 0.0592±0.00 | 0.4585±0.01 | 0.0876±0.02 | 1.1513±0.27| -0.0795±0.02 |
| DoubleEnsemble (Chuheng Zhang, et al.) | Alpha360 | 0.0407±0.00| 0.3053±0.00 | 0.0490±0.00 | 0.3840±0.00 | 0.0380±0.02 | 0.5000±0.21 | -0.0984±0.02 |
| TabNet (Sercan O. Arik, et al.)| Alpha360 | 0.0192±0.00 | 0.1401±0.00| 0.0291±0.00 | 0.2163±0.00 | -0.0258±0.00 | -0.2961±0.00| -0.1429±0.00 |
| TCTS (Xueqing Wu, et al.)| Alpha360 | 0.0485±0.00 | 0.3689±0.04| 0.0586±0.00 | 0.4669±0.02 | 0.0816±0.02 | 1.1572±0.30| -0.0689±0.02 |

## Alpha158 dataset
| Model Name | Dataset | IC | ICIR | Rank IC | Rank ICIR | Annualized Return | Information Ratio | Max Drawdown |
|---|---|---|---|---|---|---|---|---|
| Linear | Alpha158 | 0.0393±0.00 | 0.2980±0.00| 0.0475±0.00 | 0.3546±0.00 | 0.0795±0.00 | 1.0712±0.00| -0.1449±0.00 |
| CatBoost (Liudmila Prokhorenkova, et al.) | Alpha158 | 0.0503±0.00 | 0.3586±0.00| 0.0483±0.00 | 0.3667±0.00 | 0.1080±0.00 | 1.1561±0.00| -0.0787±0.00 |
| XGBoost (Tianqi Chen, et al.) | Alpha158 | 0.0481±0.00 | 0.3659±0.00| 0.0495±0.00 | 0.4033±0.00 | 0.1111±0.00 | 1.2915±0.00| -0.0893±0.00 |
| LightGBM (Guolin Ke, et al.) | Alpha158 | 0.0475±0.00 | 0.3979±0.00| 0.0485±0.00 | 0.4123±0.00 | 0.1143±0.00 | 1.2744±0.00| -0.0800±0.00 |
| MLP | Alpha158 | 0.0358±0.00 | 0.2738±0.03| 0.0425±0.00 | 0.3221±0.01 | 0.0836±0.02 | 1.0323±0.25| -0.1127±0.02 |
| TFT (Bryan Lim, et al.) | Alpha158 (with selected 20 features) | 0.0343±0.00 | 0.2071±0.02| 0.0107±0.00 | 0.0660±0.02 | 0.0623±0.02 | 0.5818±0.20| -0.1762±0.01 |
| GRU (Kyunghyun Cho, et al.) | Alpha158 (with selected 20 features) | 0.0311±0.00 | 0.2418±0.04| 0.0425±0.00 | 0.3434±0.02 | 0.0330±0.02 | 0.4805±0.30| -0.1021±0.02 |
| LSTM (Sepp Hochreiter, et al.) | Alpha158 (with selected 20 features) | 0.0312±0.00 | 0.2394±0.04| 0.0418±0.00 | 0.3324±0.03 | 0.0298±0.02 | 0.4198±0.33| -0.1348±0.03 |
| ALSTM (Yao Qin, et al.) | Alpha158 (with selected 20 features) | 0.0385±0.01 | 0.3022±0.06| 0.0478±0.00 | 0.3874±0.04 | 0.0486±0.03 | 0.7141±0.45| -0.1088±0.03 |
| GATs (Petar Velickovic, et al.) | Alpha158 (with selected 20 features) | 0.0349±0.00 | 0.2511±0.01| 0.0457±0.00 | 0.3537±0.01 | 0.0578±0.02 | 0.8221±0.25| -0.0824±0.02 |
| DoubleEnsemble (Chuheng Zhang, et al.) | Alpha158 | 0.0544±0.00 | 0.4338±0.01 | 0.0523±0.00 | 0.4257±0.01 | 0.1253±0.01 | 1.4105±0.14 | -0.0902±0.01 |
| TabNet (Sercan O. Arik, et al.)| Alpha158 | 0.0383±0.00 | 0.3414±0.00| 0.0388±0.00 | 0.3460±0.00 | 0.0226±0.00 | 0.2652±0.00| -0.1072±0.00 |

- The selected 20 features are based on the feature importance of a lightgbm-based model.
- The base model of DoubleEnsemble is LGBM.
