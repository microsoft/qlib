# Introduction
Due to the non-stationary nature of the environment of the financial market, the data distribution may change in different periods, which makes the performance of models build on training data decays in the future test data.
So adapting the forecasting models/strategies to market dynamics is very important to the model/strategies' performance.

The table below shows the performances of different solutions on different forecasting models.

## Alpha158 Dataset
Here is the [crowd sourced version of qlib data](data_collector/crowd_source/README.md): https://github.com/chenditc/investment_data/releases
```bash
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
mkdir -p ~/.qlib/qlib_data/cn_data
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=2
rm -f qlib_bin.tar.gz
```

| Model Name       | Dataset | IC | ICIR | Rank IC | Rank ICIR | Annualized Return | Information Ratio | Max Drawdown |
|------------------|---------|------|------|---------|-----------|-------------------|-------------------|--------------|
| RR[Linear]       |Alpha158 |0.0945|0.5989|0.1069   |0.6495     |0.0857             |1.3682             |-0.0986       |
| DDG-DA[Linear]   |Alpha158 |0.0983|0.6157|0.1108   |0.6646     |0.0764             |1.1904             |-0.0769       |
| RR[LightGBM]     |Alpha158 |0.0816|0.5887|0.0912   |0.6263     |0.0771             |1.3196             |-0.0909       |
| DDG-DA[LightGBM] |Alpha158 |0.0878|0.6185|0.0975   |0.6524     |0.1261             |2.0096             |-0.0744       |

- The label horizon of the `Alpha158` dataset is set to 20.
- The rolling time intervals are set to 20 trading days.
- The test rolling periods are from January 2017 to August 2020.
- The results are based on the crowd-sourced version. The Yahoo version of qlib data does not contain `VWAP`, so all related factors are missing and filled with 0, which leads to a rank-deficient matrix (a matrix does not have full rank) and makes lower-level optimization of DDG-DA can not be solved.
