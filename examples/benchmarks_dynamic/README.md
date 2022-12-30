# Introduction
Due to the non-stationary nature of the environment of the financial market, the data distribution may change in different periods, which makes the performance of models build on training data decays in the future test data.
So adapting the forecasting models/strategies to market dynamics is very important to the model/strategies' performance.

The table below shows the performances of different solutions on different forecasting models.

## Alpha158 Dataset
- The label horizon of the `Alpha158` dataset is set to 20.
- The rolling time intervals are set to 20 trading days.
- The test rolling periods are from January 2017 to August 2020.

### Internal Data
| Model Name       | Dataset | IC | ICIR | Rank IC | Rank ICIR | Annualized Return | Information Ratio | Max Drawdown |
|------------------|---------|----|------|---------|-----------|-------------------|-------------------|--------------|
| RR[Linear]       |Alpha158 |0.088|0.570|0.102    |0.622      |0.077              |1.175              |-0.086        |
| DDG-DA[Linear]   |Alpha158 |0.093|0.622|0.106    |0.670      |0.085              |1.213              |-0.093        |
| RR[LightGBM]     |Alpha158 |0.079|0.566|0.088    |0.592      |0.075              |1.226              |-0.096        |
| DDG-DA[LightGBM] |Alpha158 |0.084|0.639|0.093    |0.664      |0.099              |1.442              |-0.071        |

### Crowd Sourced Data
Here is the [crowd sourced version of qlib data](data_collector/crowd_source/README.md): https://github.com/chenditc/investment_data/releases
```bash
wget https://github.com/chenditc/investment_data/releases/download/20220720/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=2
```
| Model Name       | Dataset | IC | ICIR | Rank IC | Rank ICIR | Annualized Return | Information Ratio | Max Drawdown |
|------------------|---------|----|------|---------|-----------|-------------------|-------------------|--------------|
| RR[Linear]       |Alpha158 |0.089|0.577|0.102    |0.627      |0.093              |1.458              |-0.073        |
| DDG-DA[Linear]   |Alpha158 |0.096|0.636|0.107    |0.677      |0.067              |0.996              |-0.091        |
| RR[LightGBM]     |Alpha158 |0.082|0.589|0.091    |0.626      |0.077              |1.320              |-0.091        |
| DDG-DA[LightGBM] |Alpha158 |0.085|0.658|0.094    |0.686      |0.115              |1.792              |-0.068        |
