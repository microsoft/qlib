# Benchmarks Performance

Here are the results of each benchmark model running on Qlib's `Alpha360` and `Alpha158` dataset with China's A shared-stock & CSI300 data respectively. The values of each metric are the mean and std calculated based on 20 runs with different random seeds.

The numbers shown below demonstrate the performance of the entire `workflow` of each model. We will update the `workflow` as well as models in the near future for better results.
<!-- 
> If you need to reproduce the results below, please use the **v1** dataset: `python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/qlib_cn_1d --region cn --version v1`
>
> In the new version of qlib, the default dataset is **v2**. Since the data is collected from the YahooFinance API (which is not very stable), the results of *v2* and *v1* may differ -->

## Alpha158 dataset

| Model Name                               | Dataset                             | IC          | ICIR        | Rank IC     | Rank ICIR   | Annualized Return | Information Ratio | Max Drawdown |
|------------------------------------------|-------------------------------------|-------------|-------------|-------------|-------------|-------------------|-------------------|--------------|
| TabNet(Sercan O. Arik, et al.)           | Alpha158                            | 0.0204±0.01 | 0.1554±0.07 | 0.0333±0.00 | 0.2552±0.05 | 0.0227±0.04       | 0.3676±0.54       | -0.1089±0.08 |
| Transformer(Ashish Vaswani, et al.)      | Alpha158                            | 0.0264±0.00 | 0.2053±0.02 | 0.0407±0.00 | 0.3273±0.02 | 0.0273±0.02       | 0.3970±0.26       | -0.1101±0.02 |
| GRU(Kyunghyun Cho, et al.)               | Alpha158(with selected 20 features) | 0.0315±0.00 | 0.2450±0.04 | 0.0428±0.00 | 0.3440±0.03 | 0.0344±0.02       | 0.5160±0.25       | -0.1017±0.02 |
| LSTM(Sepp Hochreiter, et al.)            | Alpha158(with selected 20 features) | 0.0318±0.00 | 0.2367±0.04 | 0.0435±0.00 | 0.3389±0.03 | 0.0381±0.03       | 0.5561±0.46       | -0.1207±0.04 |
| Localformer(Juyong Jiang, et al.)        | Alpha158                            | 0.0356±0.00 | 0.2756±0.03 | 0.0468±0.00 | 0.3784±0.03 | 0.0438±0.02       | 0.6600±0.33       | -0.0952±0.02 |
| SFM(Liheng Zhang, et al.)                | Alpha158                            | 0.0379±0.00 | 0.2959±0.04 | 0.0464±0.00 | 0.3825±0.04 | 0.0465±0.02       | 0.5672±0.29       | -0.1282±0.03 |
| ALSTM (Yao Qin, et al.)                  | Alpha158(with selected 20 features) | 0.0362±0.01 | 0.2789±0.06 | 0.0463±0.01 | 0.3661±0.05 | 0.0470±0.03       | 0.6992±0.47       | -0.1072±0.03 |
| GATs (Petar Velickovic, et al.)          | Alpha158(with selected 20 features) | 0.0349±0.00 | 0.2511±0.01 | 0.0462±0.00 | 0.3564±0.01 | 0.0497±0.01       | 0.7338±0.19       | -0.0777±0.02 |
| TRA(Hengxu Lin, et al.)                  | Alpha158(with selected 20 features) | 0.0404±0.00 | 0.3197±0.05 | 0.0490±0.00 | 0.4047±0.04 | 0.0649±0.02       | 1.0091±0.30       | -0.0860±0.02 |
| Linear                                   | Alpha158                            | 0.0397±0.00 | 0.3000±0.00 | 0.0472±0.00 | 0.3531±0.00 | 0.0692±0.00       | 0.9209±0.00       | -0.1509±0.00 |
| TRA(Hengxu Lin, et al.)                  | Alpha158                            | 0.0440±0.00 | 0.3535±0.05 | 0.0540±0.00 | 0.4451±0.03 | 0.0718±0.02       | 1.0835±0.35       | -0.0760±0.02 |
| CatBoost(Liudmila Prokhorenkova, et al.) | Alpha158                            | 0.0481±0.00 | 0.3366±0.00 | 0.0454±0.00 | 0.3311±0.00 | 0.0765±0.00       | 0.8032±0.01       | -0.1092±0.00 |
| XGBoost(Tianqi Chen, et al.)             | Alpha158                            | 0.0498±0.00 | 0.3779±0.00 | 0.0505±0.00 | 0.4131±0.00 | 0.0780±0.00       | 0.9070±0.00       | -0.1168±0.00 |
| TFT (Bryan Lim, et al.)                  | Alpha158(with selected 20 features) | 0.0358±0.00 | 0.2160±0.03 | 0.0116±0.01 | 0.0720±0.03 | 0.0847±0.02       | 0.8131±0.19       | -0.1824±0.03 |
| MLP                                      | Alpha158                            | 0.0376±0.00 | 0.2846±0.02 | 0.0429±0.00 | 0.3220±0.01 | 0.0895±0.02       | 1.1408±0.23       | -0.1103±0.02 |
| LightGBM(Guolin Ke, et al.)              | Alpha158                            | 0.0448±0.00 | 0.3660±0.00 | 0.0469±0.00 | 0.3877±0.00 | 0.0901±0.00       | 1.0164±0.00       | -0.1038±0.00 |
| DoubleEnsemble(Chuheng Zhang, et al.)    | Alpha158                            | 0.0544±0.00 | 0.4340±0.00 | 0.0523±0.00 | 0.4284±0.01 | 0.1168±0.01       | 1.3384±0.12       | -0.1036±0.01 |



## Alpha360 dataset

| Model Name                                | Dataset  | IC          | ICIR        | Rank IC     | Rank ICIR   | Annualized Return | Information Ratio | Max Drawdown |
|-------------------------------------------|----------|-------------|-------------|-------------|-------------|-------------------|-------------------|--------------|
| Transformer(Ashish Vaswani, et al.)       | Alpha360 | 0.0114±0.00 | 0.0716±0.03 | 0.0327±0.00 | 0.2248±0.02 | -0.0270±0.03      | -0.3378±0.37      | -0.1653±0.05 |
| TabNet(Sercan O. Arik, et al.)            | Alpha360 | 0.0099±0.00 | 0.0593±0.00 | 0.0290±0.00 | 0.1887±0.00 | -0.0369±0.00      | -0.3892±0.00      | -0.2145±0.00 |
| MLP                                       | Alpha360 | 0.0273±0.00 | 0.1870±0.02 | 0.0396±0.00 | 0.2910±0.02 | 0.0029±0.02       | 0.0274±0.23       | -0.1385±0.03 |
| Localformer(Juyong Jiang, et al.)         | Alpha360 | 0.0404±0.00 | 0.2932±0.04 | 0.0542±0.00 | 0.4110±0.03 | 0.0246±0.02       | 0.3211±0.21       | -0.1095±0.02 |
| CatBoost((Liudmila Prokhorenkova, et al.) | Alpha360 | 0.0378±0.00 | 0.2714±0.00 | 0.0467±0.00 | 0.3659±0.00 | 0.0292±0.00       | 0.3781±0.00       | -0.0862±0.00 |
| XGBoost(Tianqi Chen, et al.)              | Alpha360 | 0.0394±0.00 | 0.2909±0.00 | 0.0448±0.00 | 0.3679±0.00 | 0.0344±0.00       | 0.4527±0.02       | -0.1004±0.00 |
| DoubleEnsemble(Chuheng Zhang, et al.)     | Alpha360 | 0.0404±0.00 | 0.3023±0.00 | 0.0495±0.00 | 0.3898±0.00 | 0.0468±0.01       | 0.6302±0.20       | -0.0860±0.01 |
| LightGBM(Guolin Ke, et al.)               | Alpha360 | 0.0400±0.00 | 0.3037±0.00 | 0.0499±0.00 | 0.4042±0.00 | 0.0558±0.00       | 0.7632±0.00       | -0.0659±0.00 |
| ALSTM (Yao Qin, et al.)                   | Alpha360 | 0.0497±0.00 | 0.3829±0.04 | 0.0599±0.00 | 0.4736±0.03 | 0.0626±0.02       | 0.8651±0.31       | -0.0994±0.03 |
| LSTM(Sepp Hochreiter, et al.)             | Alpha360 | 0.0448±0.00 | 0.3474±0.04 | 0.0549±0.00 | 0.4366±0.03 | 0.0647±0.03       | 0.8963±0.39       | -0.0875±0.02 |
| GRU(Kyunghyun Cho, et al.)                | Alpha360 | 0.0493±0.00 | 0.3772±0.04 | 0.0584±0.00 | 0.4638±0.03 | 0.0720±0.02       | 0.9730±0.33       | -0.0821±0.02 |
| GATs (Petar Velickovic, et al.)           | Alpha360 | 0.0476±0.00 | 0.3508±0.02 | 0.0598±0.00 | 0.4604±0.01 | 0.0824±0.02       | 1.1079±0.26       | -0.0894±0.03 |
| TCTS(Xueqing Wu, et al.)                  | Alpha360 | 0.0508±0.00 | 0.3931±0.04 | 0.0599±0.00 | 0.4756±0.03 | 0.0893±0.03       | 1.2256±0.36       | -0.0857±0.02 |
| TRA(Hengxu Lin, et al.)                   | Alpha360 | 0.0485±0.00 | 0.3787±0.03 | 0.0587±0.00 | 0.4756±0.03 | 0.0920±0.03       | 1.2789±0.42       | -0.0834±0.02 |

- The selected 20 features are based on the feature importance of a lightgbm-based model.
- The base model of DoubleEnsemble is LGBM.
- The base model of TCTS is GRU.
