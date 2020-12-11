# Benchmarks Performance

Here are the results of each benchmark model running on Qlib's `Alpha360` and `Alpha158` dataset with China's A shared-stock & CSI300 data respectively. The values of each metric are the mean and std calculated based on 10 runs.

The numbers shown below demonstrate the performance of the entire `workflow` of each model. We will update the `workflow` as well as models in the near future for better results.

## Alpha360 dataset
| Model Name | Dataset | IC | ICIR | Rank IC | Rank ICIR | Annualized Return | Information Ratio | Max Drawdown |
|---|---|---|---|---|---|---|---|---|
| Linear | Alpha360 | 0.0150±0.00 | 0.1049±0.00| 0.0284±0.00 | 0.1970±0.00 | -0.0655±0.00 | -0.6985±0.00| -0.2961±0.00 |
| CatBoost (Liudmila Prokhorenkova, et al.) | Alpha360 | 0.0397±0.00 | 0.2878±0.00| 0.0470±0.00 | 0.3703±0.00 | 0.0342±0.00 | 0.4092±0.00| -0.1057±0.00 |
| XGBoost (Tianqi Chen, et al.) | Alpha360 | 0.0400±0.00 | 0.3031±0.00| 0.0461±0.00 | 0.3862±0.00 | 0.0528±0.00 | 0.6307±0.00| -0.1113±0.00 |
| LightGBM (Guolin Ke, et al.) | Alpha360 | 0.0399±0.00 | 0.3075±0.00| 0.0492±0.00 | 0.4019±0.00 | 0.0323±0.00 | 0.4370±0.00| -0.0917±0.00 |
| MLP | Alpha360 | 0.0253±0.01 | 0.1954±0.05| 0.0329±0.00 | 0.2687±0.04 | 0.0161±0.01 | 0.1989±0.19| -0.1275±0.03 |
| GRU (Kyunghyun Cho, et al.) | Alpha360 | 0.0503±0.01 | 0.3946±0.06| 0.0588±0.00 | 0.4737±0.05 | 0.0799±0.02 | 1.0940±0.26| -0.0810±0.03 |
| LSTM (Sepp Hochreiter, et al.) | Alpha360 | 0.0466±0.01 | 0.3644±0.06| 0.0555±0.00 | 0.4451±0.04 | 0.0783±0.05 | 1.0539±0.65| -0.0844±0.03 |
| ALSTM (Yao Qin, et al.) | Alpha360 | 0.0472±0.00 | 0.3558±0.04| 0.0577±0.00 | 0.4522±0.04 | 0.0522±0.02 | 0.7090±0.32| -0.1059±0.03 |
| GATs (Petar Velickovic, et al.) | Alpha360 | 0.0480±0.00 | 0.3555±0.02| 0.0598±0.00 | 0.4616±0.01 | 0.0857±0.03 | 1.1317±0.42| -0.0917±0.01 |

## Alpha158 dataset
| Model Name | Dataset | IC | ICIR | Rank IC | Rank ICIR | Annualized Return | Information Ratio | Max Drawdown |
|---|---|---|---|---|---|---|---|---|
| Linear | Alpha158 | 0.0393±0.00 | 0.2980±0.00| 0.0475±0.00 | 0.3546±0.00 | 0.0795±0.00 | 1.0712±0.00| -0.1449±0.00 |
| CatBoost (Liudmila Prokhorenkova, et al.) | Alpha158 | 0.0503±0.00 | 0.3586±0.00| 0.0483±0.00 | 0.3667±0.00 | 0.1080±0.00 | 1.1567±0.00| -0.0787±0.00 |
| XGBoost (Tianqi Chen, et al.) | Alpha158 | 0.0481±0.00 | 0.3659±0.00| 0.0495±0.00 | 0.4033±0.00 | 0.1111±0.00 | 1.2915±0.00| -0.0893±0.00 |
| LightGBM (Guolin Ke, et al.) | Alpha158 | 0.0475±0.00 | 0.3979±0.00| 0.0485±0.00 | 0.4123±0.00 | 0.1143±0.00 | 1.2744±0.00| -0.0800±0.00 |
| MLP | Alpha158 | 0.0363±0.00 | 0.2770±0.02| 0.0421±0.00 | 0.3167±0.01 | 0.0856±0.01 | 1.0397±0.12| -0.1134±0.01 |
| TFT (Bryan Lim, et al.) | Alpha158 (with selected 20 features) | 0.0344±0.00 | 0.2071±0.02| 0.0103±0.00 | 0.0632±0.01 | 0.0638±0.00 | 0.5845±0.08| -0.1754±0.02 |
| GRU (Kyunghyun Cho, et al.) | Alpha158 (with selected 20 features) | 0.0302±0.00 | 0.2353±0.03| 0.0411±0.00 | 0.3309±0.03 | 0.0302±0.02 | 0.4353±0.28| -0.1140±0.02 |
| LSTM (Sepp Hochreiter, et al.) | Alpha158 (with selected 20 features) | 0.0359±0.01 | 0.2774±0.06| 0.0448±0.01 | 0.3597±0.05 | 0.0402±0.03 | 0.5743±0.41| -0.1152±0.03 |
| ALSTM (Yao Qin, et al.) | Alpha158 (with selected 20 features) | 0.0329±0.01 | 0.2465±0.07| 0.0450±0.01 | 0.3485±0.06 | 0.0288±0.04 | 0.4163±0.50| -0.1269±0.04 |
| GATs (Petar Velickovic, et al.) | Alpha158 (with selected 20 features) | 0.0349±0.00 | 0.2526±0.01| 0.0454±0.00 | 0.3531±0.01 | 0.0561±0.01 | 0.7992±0.19| -0.0751±0.02 |