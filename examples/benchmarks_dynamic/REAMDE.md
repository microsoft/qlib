# Introduction

Due to the non-stationary nature of the environment, the data distribution may change in different periods. However, there are still many cases that some underlying factors of environment evolution are predictable, making it possible to model the future trend of the streaming data.

Modeling the dynamics of the market is a very important problem in Quant research. On this page, we first provide the framework of periodically Rolling Retrain (RR) forecasting models so that forecasting models can learn up-to-date distributions when doing the forecasting process. Moreover, we implement a `Meta Model` module, `DDG-DA`, which effectively forecasts the evolution of data distribution and improves the performance of the RR forecasting models.

The table below shows the performances of the original RR and `DDG-DA` on different forecasting models.

## Alpha158 dataset

| Model Name       | Dataset | IC | ICIR | Rank IC | Rank ICIR | Annualized Return | Information Ratio | Max Drawdown |
|------------------|---------|----|------|---------|-----------|-------------------|-------------------|--------------|
| RR[Linear]       |Alpha158 |0.088|0.570|0.102    |0.622      |0.077              |1.175              |-0.086        |
| DDG-DA[Linear]   |Alpha158 |0.093|0.622|0.106    |0.670      |0.085              |1.213              |-0.093        |
| RR[LightGBM]     |Alpha158 |0.079|0.566|0.088    |0.592      |0.075              |1.226              |-0.096        |
| DDG-DA[LightGBM] |Alpha158 |0.084|0.639|0.093    |0.664      |0.099              |1.442              |-0.071        |

- The label horizon of the `Alpha158` dataset is set to 20.
- The rolling time intervals are set to 20 trading days.
- The test rolling periods are from January 2017 to August 2020.