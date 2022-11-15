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

 `
 python ../../scripts/data_collector/yahoo/collector.py update_data_to_bin
 `
 `
 --qlib_data_1d_dir ~/.qlib/qlib_data/us_selected_data --trading_date 2022-10-31 --region US
 `
  
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
 
 
  
  

