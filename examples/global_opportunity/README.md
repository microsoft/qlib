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

  Totally conducted 32 experiments with 2 types of datasets (Alpha158, Alpha360), 4 forecasting model algorithm (LightGBM, DoubleEnsemble, Gated Recurrent Unit Encoder-Decoder, Temporal Routing Adapter with Attention Long-Short Memory) and 2 rebalancing strategy (Top K Drop, Enhanced Indexing).  

  |**Rebalancing Strategy**|**Forecasting Model**|**Alpha Dataset**|**MSE**|**MAE**|**IC**|**IR**|**SR**|
  | :- | :- | :- | :-: | :-: | :-: | :-: | :-: |
  |Top K Drop<br>(k=10, d=1)|LightGBM|Alpha158|0.157|0.318|0.055|0.164|0.56|
  |||Alpha360|0.158|0.32|0.048|0.131|0.44|
  ||Double Ensemble|Alpha158|0.156|0.316|0.061|0.169|0.75|
  |||Alpha360|0.157|0.317|0.057|0.164|0.14|
  ||GRU|Alpha158|0.157|0.315|0.056|0.161|0.6|
  |||Alpha360|0.159|0.321|0.042|0.117|0.35|
  ||TRA ALSTM|Alpha158|0.158|0.319|0.05|0.152|0.68|
  |||Alpha360|0.161|0.322|0.032|0.101|0.46|
  |**Enhance Indexing**|LightGBM|Alpha158|0.157|0.318|0.055|0.164|0.59|
  |||Alpha360|0.158|0.32|0.048|0.131|0.68|
  ||**Double Ensemble**|**Alpha158**|**0.156**|**0.315**|**0.062**|**0.17**|**0.65**|
  |||Alpha360|0.158|0.318|0.052|0.153|0.71|
  ||GRU|Alpha158|0.158|0.317|0.05|0.139|0.55|
  |||Alpha360|0.158|0.318|0.051|0.153|0.55|
  ||TRA ALSTM|Alpha158|0.162|0.324|0.028|0.097|0.58|
  |||Alpha360|0.161|0.322|0.032|0.101|0.57|

  |**Rebalancing Strategy**|**Forecasting Model**|**Alpha Dataset**|**Without Cost**|**With Cost**|
  | :- | :- | :- | :-: | :-: |
  ||||**AR**|**AV**|**MDD**|**AR**|**AV**|**MDD**|
  |Benchmark (SPY)|10.57%|25.08%|38.16%|10.27%|25.08%|38.16%|
  |Top K Drop<br>(k=10, d=1)|LightGBM|Alpha158|16.16%|23.63%|33.09%|11.70%|23.62%|33.48%|
  |||Alpha360|10.65%|17.45%|29.58%|5.94%|17.44%|32.63%|
  ||Double Ensemble|Alpha158|21.82%|25.22%|35.24%|**17.25%**|25.21%|35.60%|
  |||Alpha360|5.85%|19.95%|32.34%|1.15%|19.94%|32.72%|
  ||GRU|Alpha158|16.71%|22.73%|36.73%|12.36%|22.72%|37.04%|
  |||Alpha360|10.96%|22.55%|32.75%|6.25%|22.53%|33.15%|
  ||TRA ALSTM|Alpha158|19.29%|24.03%|36.62%|14.84%|24.02%|36.98%|
  |||Alpha360|13.44%|22.89%|38.20%|8.61%|22.88%|38.55%|
  |**Enhanced Indexing**|LightGBM|Alpha158|10.28%|12.34%|12.30%|8.90%|12.33%|14.09%|
  |||Alpha360|11.00%|11.85%|11.66%|9.31%|11.85%|14.19%|
  ||**Double Ensemble**|**Alpha158**|**11.00%**|**12.34%**|**12.33%**|**9.98%**|**12.33%**|**13.32%**|
  |||Alpha360|11.35%|11.73%|11.21%|9.46%|11.73%|13.12%|
  ||GRU|Alpha158|9.70%|12.30%|13.51%|8.76%|12.29%|14.23%|
  |||Alpha360|9.64%|12.13%|13.46%|7.90%|12.13%|15.69%|
  ||TRA ALSTM|Alpha158|10.10%|12.19%|12.05%|8.26%|12.18%|14.37%|
  |||Alpha360|10.00%|12.19%|12.23%|8.36%|12.18%|12.53%|

  The obvious inferences form the evaluation metrics of the experiments detailed in the table above is that the Double Ensemble model with Alpha158 dataset has produced portfolio performance that can consistently beat the benchmark index S&P 500 when using the simple TopK Drop Rebalancing Strategy (generated an excess return of **6.98%** annually) but it failed to resist the market downturn whereas the same (Double Ensemble model with Alpha158 dataset) has shown good downside protection (reduced the maximum drawdown over **25%** during the Covid Crisis) as well as a comparable performance in case using the Enhanced Indexing Strategy.

  Information coefficient of the combination Alpha158 Dataset, Double Ensemble Model & Enhanced Indexing Strategy also stands at **0.062** which is the best out of all the experiments. Generally speaking, many portfolio managers would view a “good” IC as 0.05 and above.

  Comparing the plots illustrated in the notebook ResearchReports.ipynb clearly indicates that using Alpha158 dataset provide more valuable signal for forecasting model leading to better information coefficient and better portfolio performance even when using the simple Top K Drop Rebalancing Strategy. Learner-based feature selection approach for effective features from the standard technical factors engineered from raw data has tremendously reduced the noise in the market data and led to the outperformance over the raw price & volume data signals provided by the Alpha360 dataset.  

  Using the Enhanced Indexing Strategy has effectively reduced the trading cost by applying the mix of active-passive rebalancing strategy i.e., the need to rebalance arises only when the rebalancer notices a reasonable difference in excess return and risk in comparison to the benchmark index otherwise it avoids rebalancing which saves frequent trading and the cost implication from it. Also, this approach has limited the maximum dropdown largely and has provided an excellent downside protection to investment portfolio over the Top K Drop Strategy.

