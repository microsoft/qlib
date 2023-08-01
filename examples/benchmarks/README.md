# Benchmarks Performance
This page lists a batch of methods designed for alpha seeking. Each method tries to give scores/predictions for all stocks each day(e.g. forecasting the future excess return of stocks). The scores/predictions of the models will be used as the mined alpha. Investing in stocks with higher scores is expected to yield more profit.  

The alpha is evaluated in two ways.
1. The correlation between the alpha and future return.
1. Constructing portfolio based on the alpha and evaluating the final total return.
   - The explanation of metrics can be found [here](https://qlib.readthedocs.io/en/latest/component/report.html#id4)

Here are the results of each benchmark model running on Qlib's `Alpha360` and `Alpha158` dataset with China's A shared-stock & CSI300 data respectively. The values of each metric are the mean and std calculated based on 20 runs with different random seeds.

The numbers shown below demonstrate the performance of the entire `workflow` of each model. We will update the `workflow` as well as models in the near future for better results.
<!-- 
> If you need to reproduce the results below, please use the **v1** dataset: `python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn --version v1`
>
> In the new version of qlib, the default dataset is **v2**. Since the data is collected from the YahooFinance API (which is not very stable), the results of *v2* and *v1* may differ -->

> NOTE:
> The backtest start from 0.8.0 is quite different from previous version. Please check out the changelog for the difference.

> NOTE:
> We have very limited resources to implement and finetune the models. We tried our best effort to fairly compare these models.  But some models may have greater potential than what it looks like in the table below.  Your contribution is highly welcomed to explore their potential.

## Results on CSI300

### Alpha158 dataset

| Model Name                               | Dataset                             | IC          | ICIR        | Rank IC     | Rank ICIR   | Annualized Return | Information Ratio | Max Drawdown |
|------------------------------------------|-------------------------------------|-------------|-------------|-------------|-------------|-------------------|-------------------|--------------|
| TCN(Shaojie Bai, et al.)                 | Alpha158                            | 0.0279±0.00 | 0.2181±0.01 | 0.0421±0.00 | 0.3429±0.01 | 0.0262±0.02       | 0.4133±0.25       | -0.1090±0.03 |
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
| DoubleEnsemble(Chuheng Zhang, et al.)    | Alpha158                            | 0.0521±0.00 | 0.4223±0.01 | 0.0502±0.00 | 0.4117±0.01 | 0.1158±0.01       | 1.3432±0.11       | -0.0920±0.01 |

### Alpha360 dataset

| Model Name                                | Dataset  | IC          | ICIR        | Rank IC     | Rank ICIR   | Annualized Return | Information Ratio | Max Drawdown |
|-------------------------------------------|----------|-------------|-------------|-------------|-------------|-------------------|-------------------|--------------|
| Transformer(Ashish Vaswani, et al.)       | Alpha360 | 0.0114±0.00 | 0.0716±0.03 | 0.0327±0.00 | 0.2248±0.02 | -0.0270±0.03      | -0.3378±0.37      | -0.1653±0.05 |
| TabNet(Sercan O. Arik, et al.)            | Alpha360 | 0.0099±0.00 | 0.0593±0.00 | 0.0290±0.00 | 0.1887±0.00 | -0.0369±0.00      | -0.3892±0.00      | -0.2145±0.00 |
| MLP                                       | Alpha360 | 0.0273±0.00 | 0.1870±0.02 | 0.0396±0.00 | 0.2910±0.02 | 0.0029±0.02       | 0.0274±0.23       | -0.1385±0.03 |
| Localformer(Juyong Jiang, et al.)         | Alpha360 | 0.0404±0.00 | 0.2932±0.04 | 0.0542±0.00 | 0.4110±0.03 | 0.0246±0.02       | 0.3211±0.21       | -0.1095±0.02 |
| CatBoost((Liudmila Prokhorenkova, et al.) | Alpha360 | 0.0378±0.00 | 0.2714±0.00 | 0.0467±0.00 | 0.3659±0.00 | 0.0292±0.00       | 0.3781±0.00       | -0.0862±0.00 |
| XGBoost(Tianqi Chen, et al.)              | Alpha360 | 0.0394±0.00 | 0.2909±0.00 | 0.0448±0.00 | 0.3679±0.00 | 0.0344±0.00       | 0.4527±0.02       | -0.1004±0.00 |
| DoubleEnsemble(Chuheng Zhang, et al.)     | Alpha360 | 0.0390±0.00 | 0.2946±0.01 | 0.0486±0.00 | 0.3836±0.01 | 0.0462±0.01       | 0.6151±0.18       | -0.0915±0.01 |
| LightGBM(Guolin Ke, et al.)               | Alpha360 | 0.0400±0.00 | 0.3037±0.00 | 0.0499±0.00 | 0.4042±0.00 | 0.0558±0.00       | 0.7632±0.00       | -0.0659±0.00 |
| TCN(Shaojie Bai, et al.)                  | Alpha360 | 0.0441±0.00 | 0.3301±0.02 | 0.0519±0.00 | 0.4130±0.01 | 0.0604±0.02       | 0.8295±0.34       | -0.1018±0.03 |
| ALSTM (Yao Qin, et al.)                   | Alpha360 | 0.0497±0.00 | 0.3829±0.04 | 0.0599±0.00 | 0.4736±0.03 | 0.0626±0.02       | 0.8651±0.31       | -0.0994±0.03 |
| LSTM(Sepp Hochreiter, et al.)             | Alpha360 | 0.0448±0.00 | 0.3474±0.04 | 0.0549±0.00 | 0.4366±0.03 | 0.0647±0.03       | 0.8963±0.39       | -0.0875±0.02 |
| ADD                                       | Alpha360 | 0.0430±0.00 | 0.3188±0.04 | 0.0559±0.00 | 0.4301±0.03 | 0.0667±0.02       | 0.8992±0.34       | -0.0855±0.02 |
| GRU(Kyunghyun Cho, et al.)                | Alpha360 | 0.0493±0.00 | 0.3772±0.04 | 0.0584±0.00 | 0.4638±0.03 | 0.0720±0.02       | 0.9730±0.33       | -0.0821±0.02 |
| AdaRNN(Yuntao Du, et al.)                 | Alpha360 | 0.0464±0.01 | 0.3619±0.08 | 0.0539±0.01 | 0.4287±0.06 | 0.0753±0.03       | 1.0200±0.40       | -0.0936±0.03 |
| GATs (Petar Velickovic, et al.)           | Alpha360 | 0.0476±0.00 | 0.3508±0.02 | 0.0598±0.00 | 0.4604±0.01 | 0.0824±0.02       | 1.1079±0.26       | -0.0894±0.03 |
| TCTS(Xueqing Wu, et al.)                  | Alpha360 | 0.0508±0.00 | 0.3931±0.04 | 0.0599±0.00 | 0.4756±0.03 | 0.0893±0.03       | 1.2256±0.36       | -0.0857±0.02 |
| TRA(Hengxu Lin, et al.)                   | Alpha360 | 0.0485±0.00 | 0.3787±0.03 | 0.0587±0.00 | 0.4756±0.03 | 0.0920±0.03       | 1.2789±0.42       | -0.0834±0.02 |
| IGMTF(Wentao Xu, et al.)                  | Alpha360 | 0.0480±0.00 | 0.3589±0.02 | 0.0606±0.00 | 0.4773±0.01 | 0.0946±0.02       | 1.3509±0.25       | -0.0716±0.02 |
| HIST(Wentao Xu, et al.)                   | Alpha360 | 0.0522±0.00 | 0.3530±0.01 | 0.0667±0.00 | 0.4576±0.01 | 0.0987±0.02       | 1.3726±0.27       | -0.0681±0.01 |
| KRNN                                      | Alpha360 | 0.0173±0.01 | 0.1210±0.06 | 0.0270±0.01 | 0.2018±0.04 | -0.0465±0.05      | -0.5415±0.62      | -0.2919±0.13 |
| Sandwich                                  | Alpha360 | 0.0258±0.00 | 0.1924±0.04 | 0.0337±0.00 | 0.2624±0.03 | 0.0005±0.03       | 0.0001±0.33       | -0.1752±0.05 |


- The selected 20 features are based on the feature importance of a lightgbm-based model.
- The base model of DoubleEnsemble is LGBM.
- The base model of TCTS is GRU.
- About the datasets
  - Alpha158 is a tabular dataset. There are less spatial relationships between different features. Each feature are carefully designed by human (a.k.a feature engineering)
  - Alpha360 contains raw price and volue data without much feature engineering. There are strong strong spatial relationships between the features in the time dimension.
- The metrics can be categorized into two
   - Signal-based evaluation:  IC, ICIR, Rank IC, Rank ICIR
      - ![equation](https://latex.codecogs.com/gif.latex?%5Ctext%7Bcorr%7D%28%5Ctextbf%7Bx%7D%2C%5Ctextbf%7By%7D%29%3D%5Cfrac%7B%5Csum_i%20%28x_i-%5Cbar%7Bx%7D%29%28y_i-%5Cbar%7By%7D%29%7D%7B%5Csqrt%7B%5Csum_i%28x_i-%5Cbar%7Bx%7D%29%5E2%5Csum_i%28y_i-%5Cbar%7By%7D%29%5E2%7D%7D)
      - ![equation](https://latex.codecogs.com/gif.latex?%5Ctext%7BIC%7D%5E%7B%28t%29%7D%20%3D%20%5Ctext%7Bcorr%7D%28%5Chat%7B%5Ctextbf%7By%7D%7D%5E%7B%28t%29%7D%2C%20%5Ctextbf%7Bret%7D%5E%7B%28t%29%7D%29)
      - ![equation](https://latex.codecogs.com/gif.latex?%5Ctext%7BICIR%7D%20%3D%20%5Cfrac%20%7B%5Ctext%7Bmean%7D%28%5Ctextbf%7BIC%7D%29%7D%20%7B%5Ctext%7Bstd%7D%28%5Ctextbf%7BIC%7D%29%7D)
      - ![equation](https://latex.codecogs.com/gif.latex?%5Ctext%7BRank%20IC%7D%5E%7B%28t%29%7D%20%3D%20%5Ctext%7Bcorr%7D%28%5Ctext%7Brank%7D%28%5Chat%7B%5Ctextbf%7By%7D%7D%5E%7B%28t%29%7D%29%2C%20%5Ctext%7Brank%7D%28%5Ctextbf%7Bret%7D%5E%7B%28t%29%7D%29%29)
      - ![equation](https://latex.codecogs.com/gif.latex?%5Ctext%7BRank%20ICIR%7D%20%3D%20%5Cfrac%20%7B%5Ctext%7Bmean%7D%28%5Ctextbf%7BRank%20IC%7D%29%7D%20%7B%5Ctext%7Bstd%7D%28%5Ctextbf%7BRankIC%7D%29%7D)
   - Portfolio-based metrics:  Annualized Return, Information Ratio, Max Drawdown

## Results on CSI500
The results on CSI500 is not complete. PR's for models on csi500 are welcome!

Transfer previous models in CSI300 to CSI500 is quite easy.  You can try models with just a few commands below.
```
cd examples/benchmarks/LightGBM
pip install -r requirements.txt

# create new config and set the benchmark to csi500
cp workflow_config_lightgbm_Alpha158.yaml workflow_config_lightgbm_Alpha158_csi500.yaml
sed -i "s/csi300/csi500/g"  workflow_config_lightgbm_Alpha158_csi500.yaml
sed -i "s/SH000300/SH000905/g"  workflow_config_lightgbm_Alpha158_csi500.yaml

# you can either run the model once
qrun workflow_config_lightgbm_Alpha158_csi500.yaml

# or run it for multiple times automatically and get the summarized results.
cd  ../../
python run_all_model.py run 3 lightgbm Alpha158 csi500  # for models with randomness.  please run it for 20 times.
```

### Alpha158 dataset
| Model Name | Dataset  | IC          | ICIR        | Rank IC     | Rank ICIR   | Annualized Return | Information Ratio | Max Drawdown |
|------------|----------|-------------|-------------|-------------|-------------|-------------------|-------------------|--------------|
| Linear     | Alpha158 | 0.0332±0.00 | 0.3044±0.00 | 0.0462±0.00 | 0.4326±0.00 | 0.0382±0.00       | 0.1723±0.00       | -0.4876±0.00 |
| MLP        | Alpha158 | 0.0229±0.01 | 0.2181±0.05 | 0.0360±0.00 | 0.3409±0.02 | 0.0043±0.02       | 0.0602±0.27       | -0.2184±0.04 |
| LightGBM   | Alpha158 | 0.0399±0.00 | 0.4065±0.00 | 0.0482±0.00 | 0.5101±0.00 | 0.1284±0.00       | 1.5650±0.00       | -0.0635±0.00 |
| CatBoost   | Alpha158 | 0.0345±0.00 | 0.2855±0.00 | 0.0417±0.00 | 0.3740±0.00 | 0.0496±0.00       | 0.5977±0.00       | -0.1496±0.00 |
| DoubleEnsemble  | Alpha158 | 0.0380±0.00 | 0.3659±0.00 | 0.0442±0.00 | 0.4324±0.00 | 0.0382±0.00       | 0.1723±0.00       | -0.4876±0.00 |

### Alpha360 dataset
| Model Name | Dataset  | IC          | ICIR        | Rank IC     | Rank ICIR   | Annualized Return | Information Ratio | Max Drawdown |
|------------|----------|-------------|-------------|-------------|-------------|-------------------|-------------------|--------------|
| MLP        | Alpha360 | 0.0258±0.00 | 0.2021±0.02 | 0.0426±0.00 | 0.3840±0.02 | 0.0022±0.02       | 0.0301±0.26       | -0.2064±0.02 |
| LightGBM   | Alpha360 | 0.0400±0.00 | 0.3605±0.00 | 0.0536±0.00 | 0.5431±0.00 | 0.0505±0.00       | 0.7658±0.02       | -0.1880±0.00 |
| CatBoost   | Alpha360 | 0.0382±0.00 | 0.3229±0.00 | 0.0489±0.00 | 0.4649±0.00 | 0.0297±0.00       | 0.4227±0.02       | -0.1499±0.01 |
| DoubleEnsemble  | Alpha360 | 0.0361±0.00 | 0.3092±0.00 | 0.0499±0.00 | 0.4793±0.00 | 0.0382±0.00       | 0.1723±0.02       | -0.4876±0.00 |

# Contributing

Your contributions to new models are highly welcome!

If you want to contribute your new models, you can follow the steps below.
1. Create a folder for your model
2. The folder contains following items(you can refer to [this example](https://github.com/microsoft/qlib/tree/main/examples/benchmarks/TCTS)).
    - `requirements.txt`: required dependencies.
    - `README.md`: a brief introduction to your models
    - `workflow_config_<model name>_<dataset>.yaml`: a configuration which can read by `qrun`. You are encouraged to run your model in all datasets.
3. You can integrate your model as a module [in this folder](https://github.com/microsoft/qlib/tree/main/qlib/contrib/model).
4. Please update your results in the above **Benchmark Tables**, e.g. [Alpha360](#alpha158-dataset), [Alpha158](#alpha158-dataset)(the values of each metric are the mean and std calculated based on **20 Runs** with different random seeds. You can accomplish the above operations through the automated [script](https://github.com/microsoft/qlib/blob/main/examples/run_all_model.py) provided by Qlib, and get the final result in the .md file. if you don't have enough computational resource, you can ask for help in the PR).
5. Update the info in the index page in the [news list](https://github.com/microsoft/qlib#newspaper-whats-new----sparkling_heart) and [model list](https://github.com/microsoft/qlib#quant-model-paper-zoo).

Finally, you can send PR for review. ([here is an example](https://github.com/microsoft/qlib/pull/1040))


# FAQ

Q: What's the difference between models with name `*.py` and `*_ts.py`?

A: Models with name `*_ts.py` are designed for `TSDatasetH` (`TSDatasetH` will create time-series automatically from tabular data).  Models with name `*.py` are designed for `DatasetH` (`DatasetH` is usually used in tabular data.  But users still can apply time-series models on tabular datasets if the columns has time-series relationships). 
