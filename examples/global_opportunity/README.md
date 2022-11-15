# Global Investment Opportunity with Intelligent Asset Allocator (with Qlib)

## Install Qlib

- Run below command to install `qlib` from local src

 ```
 pip install ../../
 ```
#####  or
- Run below command to install `qlib from repository`

 ```
 pip install pyqlib
 ```

## Data Preparation

- Run below command to download and prepare the data from Yahoo Finance using data collector

 ```
 python ../../scripts/data_collector/yahoo/collector.py update_data_to_bin
 ```
 with parameters`--qlib_data_1d_dir ~/.qlib/qlib_data/us_selected_data --trading_date 2022-10-31 --region US`
  
- Run below command to prepare the risk data for EnhancesIndexingStrategy

 ```
 python prepare_riskdata.py
 ```

## Execute the experiments

----------------------------------
### LightGBM
----------------------------------
- Run LightGBM Model with Alpha158 dataset & EnhancedIndexingStrategy (copy & update the experiment Id in the ResearchReports for plotting the results)
 ```
 lightgbm_alpha158_eis.yaml
 ```
- Run LightGBM Model with Alpha158 dataset & TopKStrategy (copy & update the experiment Id in the ResearchReports for plotting the results)
 ```
 lightgbm_alpha158_topK.yaml
 ```

- Run LightGBM Model with Alpha360 dataset & EnhancedIndexingStrategy (copy & update the experiment Id in the ResearchReports for plotting the results)
 ```
 lightgbm_alpha360_eis.yaml
 ```
- Run LightGBM Model with Alpha360 dataset & TopKStrategy (copy & update the experiment Id in the ResearchReports for plotting the results)
 ```
 lightgbm_alpha360_topK.yaml
 ```

----------------------------------
### DounleEnsemble
----------------------------------
 - Run DounleEnsemble Model with Alpha158 dataset & EnhancedIndexingStrategy (copy & update the experiment Id in the ResearchReports for plotting the results)
 ```
 densemble_alpha158_eis.yaml
 ```
- Run DounleEnsemble Model with Alpha158 dataset & TopKStrategy (copy & update the experiment Id in the ResearchReports for plotting the results)
 ```
 densemble_alpha158_topK.yaml
 ```

- Run DounleEnsemble Model with Alpha360 dataset & EnhancedIndexingStrategy (copy & update the experiment Id in the ResearchReports for plotting the results)
 ```
 densemble_alpha360_eis.yaml
 ```
- Run DounleEnsemble Model with Alpha360 dataset & TopKStrategy (copy & update the experiment Id in the ResearchReports for plotting the results)
 ```
 densemble_alpha360_topK.yaml
 ```

----------------------------------
### Gated Recurrent Unit (GRU)
----------------------------------
 - Run Gated Recurrent Unit (GRU) Model with Alpha158 dataset & EnhancedIndexingStrategy (copy & update the experiment Id in the ResearchReports for plotting the results)
 ```
 gru_alpha158_eis.yaml
 ```
- Run Gated Recurrent Unit (GRU) Model with Alpha158 dataset & TopKStrategy (copy & update the experiment Id in the ResearchReports for plotting the results)
 ```
 gru_alpha158_topK.yaml
 ```

- Run Gated Recurrent Unit (GRU) Model with Alpha360 dataset & EnhancedIndexingStrategy (copy & update the experiment Id in the ResearchReports for plotting the results)
 ```
 gru_alpha360_eis.yaml
 ```
- Run Gated Recurrent Unit Model (GRU) with Alpha360 dataset & TopKStrategy (copy & update the experiment Id in the ResearchReports for plotting the results)
 ```
 gru_alpha360_topK.yaml
 ```
 
----------------------------------
###  Temporal Routing Adaptor (TRA) with ALSTM 
----------------------------------
 - Run Temporal Routing Adaptor (TRA) with ALSTM Model with Alpha158 dataset & EnhancedIndexingStrategy (copy & update the experiment Id in the ResearchReports for plotting the results)
 ```
 tra_lstm_alpha158_eis.yaml
 ```
- Run Temporal Routing Adaptor (TRA) with ALSTM Model with Alpha158 dataset & TopKStrategy (copy & update the experiment Id in the ResearchReports for plotting the results)
 ```
 tra_lstm_alpha158_topK.yaml
 ```

- Run Temporal Routing Adaptor (TRA) with ALSTM Model with Alpha360 dataset & EnhancedIndexingStrategy (copy & update the experiment Id in the ResearchReports for plotting the results)
 ```
 tra_lstm_alpha360_eis.yaml
 ```
- Run Temporal Routing Adaptor (TRA) with ALSTM with Alpha360 dataset & TopKStrategy (copy & update the experiment Id in the ResearchReports for plotting the results)
 ```
 tra_lstm_alpha360_topK.yaml
 ```
 ----------------------------------
 
 ## Compare the metrics and results of the experiments
 
 - Run the `ResearchReports.ipynb` in the jyputer notebook to view compare the results of experiments conducted
 
 |       **_Experiments_**      | _Squared Error_ | _Absolute Error_ | _Information Coefficient_ | _Information Ratio_ | _Annualized Return_ | _Annualized Volatility_ | _Sharpe Ratio_ | _Max Drawdown_ | _Annualized Return (with cost)_ | _Annualized Volatility (with cost)_ | _Max Drawdown (with cost)_ |
|:----------------------------:|----------------:|-----------------:|--------------------------:|--------------------:|--------------------:|------------------------:|---------------:|---------------:|--------------------------------:|------------------------------------:|---------------------------:|
|   _LightGBM_Alpha158_TopK_   |           0.157 |            0.318 |                     0.055 |               0.164 |              16.16% |                  23.63% |           0.56 |         33.09% |                           11.7% |                              23.62% |                     33.48% |
|   _LightGBM_Alpha360_TopK_   |           0.158 |             0.32 |                     0.048 |               0.131 |              10.65% |                  17.45% |           0.44 |         29.58% |                           5.94% |                              17.44% |                     32.63% |
|   _DEnsemble_Alpha158_TopK_  |           0.156 |            0.316 |                     0.061 |               0.169 |              21.82% |                  25.22% |           0.75 |         35.24% |                      **17.25%** |                              25.21% |                      35.6% |
|   _DEnsemble_Alpha360_TopK_  |           0.157 |            0.317 |                     0.057 |               0.164 |               5.85% |                  19.95% |           0.14 |         32.34% |                           1.15% |                              19.94% |                     32.72% |
|      _GRU_Alpha158_TopK_     |           0.157 |            0.315 |                     0.056 |               0.161 |              16.71% |                  22.73% |           0.60 |         36.73% |                          12.36% |                              22.72% |                     37.04% |
|      _GRU_Alpha360_TopK_     |           0.159 |            0.321 |                     0.042 |               0.117 |              10.96% |                  22.55% |           0.35 |         32.75% |                           6.25% |                              22.53% |                     33.15% |
|   _TRA_LSTM_Alpha158_TopK_   |           0.158 |            0.319 |                      0.05 |               0.152 |              19.29% |                  24.03% |           0.68 |         36.62% |                          14.84% |                              24.02% |                     36.98% |
|   _TRA_LSTM_Alpha360_TopK_   |           0.161 |            0.322 |                     0.032 |               0.101 |              13.44% |                  22.89% |           0.46 |          38.2% |                           8.61% |                              22.88% |                     38.55% |
|    _LightGBM_Alpha158_EIS_   |           0.157 |            0.318 |                     0.055 |               0.164 |              10.28% |                  12.34% |           0.59 |          12.3% |                            8.9% |                              12.33% |                     14.09% |
|    _LightGBM_Alpha360_EIS_   |           0.158 |             0.32 |                     0.048 |               0.131 |               11.0% |                  11.85% |           0.68 |         11.66% |                           9.31% |                              11.85% |                     14.19% |
| **_DEnsemble_Alpha158_EIS_** |       **0.156** |        **0.315** |                 **0.062** |            **0.17** |           **11.0%** |              **12.34%** |       **0.65** |     **12.33%** |                       **9.98%** |                          **12.33%** |                 **13.32%** |
|   _DEnsemble_Alpha360_EIS_   |           0.158 |            0.318 |                     0.052 |               0.153 |              11.35% |                  11.73% |           0.71 |         11.21% |                           9.46% |                              11.73% |                     13.12% |
|      _GRU_Alpha158_EIS_      |           0.158 |            0.317 |                      0.05 |               0.139 |                9.7% |                   12.3% |           0.55 |         13.51% |                           8.76% |                              12.29% |                     14.23% |
|      _GRU_Alpha360_EIS_      |           0.158 |            0.318 |                     0.051 |               0.153 |               9.64% |                  12.13% |           0.55 |         13.46% |                            7.9% |                              12.13% |                     15.69% |
|    _TRA_LSTM_Alpha158_EIS_   |           0.162 |            0.324 |                     0.028 |               0.097 |               10.1% |                  12.19% |           0.58 |         12.05% |                           8.26% |                              12.18% |                     14.37% |
|    _TRA_LSTM_Alpha360_EIS_   |           0.161 |            0.322 |                     0.032 |               0.101 |               10.0% |                  12.19% |           0.57 |         12.23% |                           8.36% |                              12.18% |                 **12.53%** |
|       _Benchmark (SPY)_      |               - |                - |                         - |                   - |              10.57% |                  25.08% |           0.30 |         38.16% |                          10.27% |                              25.08% |                     38.16% |
 
  
  

