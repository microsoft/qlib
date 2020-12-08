# Benchmarks Performance

Here are the results of each benchmark model running on Qlib's `Alpha360` and `Alpha158` dataset with China's A shared-stock & CSI300 data respectively. The values of each metric are the mean and std calculated based on 10 runs.

The numbers shown below demonstrate the performance of the entire `workflow` of each model. We will update the `workflow` as well as models in the near future for better results.

## Alpha360 dataset
| Model Name | Dataset | IC | ICIR | Rank IC | Rank ICIR | Annualized Return | Information Ratio | Max Drawdown |
|---|---|---|---|---|---|---|---|---|
| Linear | Alpha360 | 0.0150±0.00 | 0.1049±0.00| 0.0284±0.00 | 0.1970±0.00 | -0.0655±0.00 | -0.6985±0.00| -0.2961±0.00 |
| CatBoost | Alpha360 | 0.0397±0.00 | 0.2878±0.00| 0.0470±0.00 | 0.3703±0.00 | 0.0342±0.00 | 0.4092±0.00| -0.1057±0.00 |
| XGBoost | Alpha360 | 0.0400±0.00 | 0.3031±0.00| 0.0461±0.00 | 0.3862±0.00 | 0.0528±0.00 | 0.6307±0.00| -0.1113±0.00 |
| LightGBM | Alpha360 | 0.0399±0.00 | 0.3075±0.00| 0.0492±0.00 | 0.4019±0.00 | 0.0323±0.00 | 0.4370±0.00| -0.0917±0.00 |
| MLP | Alpha360 | 0.0253±0.01 | 0.1954±0.05| 0.0329±0.00 | 0.2687±0.04 | 0.0161±0.01 | 0.1989±0.19| -0.1275±0.03 |
| GRU | Alpha360 | 0.0503±0.01 | 0.3946±0.06| 0.0588±0.00 | 0.4737±0.05 | 0.0799±0.02 | 1.0940±0.26| -0.0810±0.03 |
| LSTM | Alpha360 | 0.0466±0.01 | 0.3644±0.06| 0.0555±0.00 | 0.4451±0.04 | 0.0783±0.05 | 1.0539±0.65| -0.0844±0.03 |
| ALSTM | Alpha360 | 0.0472±0.00 | 0.3558±0.04| 0.0577±0.00 | 0.4522±0.04 | 0.0522±0.02 | 0.7090±0.32| -0.1059±0.03 |
| GATs | Alpha360 | 0.0480±0.00 | 0.3555±0.02| 0.0598±0.00 | 0.4616±0.01 | 0.0857±0.03 | 1.1317±0.42| -0.0917±0.01 |

## Alpha158 dataset
| Model Name | Dataset | IC | ICIR | Rank IC | Rank ICIR | Annualized Return | Information Ratio | Max Drawdown |
|---|---|---|---|---|---|---|---|---|
| Linear | Alpha158 | 0.0393±0.00 | 0.2980±0.00| 0.0475±0.00 | 0.3546±0.00 | 0.0795±0.00 | 1.0712±0.00| -0.1449±0.00 |
| CatBoost | Alpha158 | 0.0503±0.00 | 0.3586±0.00| 0.0483±0.00 | 0.3667±0.00 | 0.1080±0.00 | 1.1567±0.00| -0.0787±0.00 |
| XGBoost | Alpha158 | 0.0481±0.00 | 0.3659±0.00| 0.0495±0.00 | 0.4033±0.00 | 0.1111±0.00 | 1.2915±0.00| -0.0893±0.00 |
| LightGBM | Alpha158 | 0.0475±0.00 | 0.3979±0.00| 0.0485±0.00 | 0.4123±0.00 | 0.1143±0.00 | 1.2744±0.00| -0.0800±0.00 |
| MLP | Alpha158 | 0.0363±0.00 | 0.2770±0.02| 0.0421±0.00 | 0.3167±0.01 | 0.0856±0.01 | 1.0397±0.12| -0.1134±0.01 |
| TFT | Alpha158 (with selected 10 features) | 0.0287±0.00 | 0.1663±0.01| 0.0016±0.00 | 0.0095±0.02 | 0.0205±0.02 | 0.1758±0.19| -0.1990±0.04 |
| GRU | Alpha158 (with selected 20 features) | 0.0313±0.00 | 0.2427±0.01 | 0.0416±0.00 | 0.3370±0.01 | 0.0335±0.01 | 0.4808±0.22 | -0.1112±0.03 |
| LSTM | Alpha158 (with selected 20 features) | 0.0337±0.01 | 0.2562±0.05 | 0.0427±0.01 | 0.3392±0.04 | 0.0269±0.06 | 0.3385±0.74 | -0.1285±0.04 |
| ALSTM | Alpha158 (with selected 20 features) | 0.0366±0.00 | 0.2803±0.04 | 0.0478±0.00 | 0.3770±0.02 | 0.0520±0.03 | 0.7115±0.30 | -0.0986±0.01 |
| GATs | Alpha158 (with selected 20 features) | 0.0355±0.00 | 0.2576±0.02 | 0.0465±0.00 | 0.3585±0.00 | 0.0509±0.02 | 0.7212±0.22 | -0.0821±0.01 |