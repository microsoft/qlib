import pandas as pd
import qlib
from qlib.utils import init_instance_by_config
from qlib.workflow.record_temp import SignalRecord
from qlib.workflow import R
from os.path import join
from portfolio_management_2_stage.utils.creating_dataset import create_dataset

instruments = ['PNW', 'MA', 'FDX', 'BEN', 'NKE', 'CSX', 'F', 'PKI', 'PHM', 'ADSK', 'JNJ', 'LMT', 'FIS', 'ITW', 'SHW', 'ADI', 
               'PRU', 'PFG', 'USB', 'GILD', 'LIN', 'FITB', 'UNM', 'NSC', 'HIG', 'AIZ', 'BDX', 'CMS', 'DVN', 'PEP', 'PGR', 'AMT', 
               'CMA', 'PWR', 'K', 'BKNG', 'ABC', 'CMI', 'COST', 'UNP', 'LEG', 'TROW', 'MMM', 'GD', 'CTSH', 'CMCSA', 'FISV', 
               'ZION', 'CL', 'HPQ', 'RF', 'CHRW', 'MAR', 'WEC', 'TPR', 'APH', 'LHX', 'CINF', 'PFE', 'JPM', 'SJM', 'HD', 'JNPR', 
               'IRM', 'RSG', 'LLY', 'KLAC', 'GIS', 'WMB', 'AIG', 'AIV', 'NRG', 'RL', 'GPC', 'XRX', 'TMO', 'UNH', 'CF', 'EOG', 
               'PEG', 'COF', 'STT', 'LH', 'GS', 'WMT', 'C', 'HST', 'OMC', 'PLD', 'PXD', 'ROST', 'SYK', 'OXY', 'DGX', 'AAPL', 
               'CTAS', 'GPS', 'IPG', 'GL', 'AXP', 'HON', 'KIM', 'ROP', 'AES', 'ES', 'PSA', 'AFL', 'KMB', 'CCL', 'ORLY', 'GWW', 
               'GE', 'EXC', 'MDT', 'SEE', 'TGT', 'TRV', 'MCHP', 'WDC', 'XOM', 'HBAN', 'APD', 'MKC', 'MS', 'HES', 'ETR', 'MMC', 
               'VNO', 'LNC', 'NEM', 'TAP', 'BAC', 'MRO', 'PEAK', 'IFF', 'AKAM', 'ALL', 'IP', 'VMC', 'FLS', 'DHR', 'MET', 'NUE', 
               'CNP', 'CAT', 'TXT', 'ED', 'CVX', 'SCHW', 'ETN', 'WM', 'WU', 'GOOG', 'MTB', 'SWK', 'RHI', 'KEY', 'AVB', 'GLW', 
               'CAG', 'AEE', 'T', 'PPG', 'ORCL', 'WELL', 'AMGN', 'SPGI', 'MU', 'CBRE', 'UAA', 'WY', 'EFX', 'IVZ', 'HAL', 'OKE', 
               'WFC', 'DUK', 'WYNN', 'WAT', 'PG', 'EA', 'NOC', 'BSX', 'NDAQ', 'LEN', 'BA', 'RTX', 'BMY', 'FMC', 'NVDA', 'WHR', 
               'YUM', 'NEE', 'AMP', 'SO', '^GSPC', 'EIX', 'CSCO', 'VFC', 'COP', 'EMR', 'ISRG', 'ZBH', 'AEP', 'AMAT', 'AZO', 'NOV', 
               'LOW', 'MSI', 'VRSN', 'TFC', 'INTU', 'SBUX', 'NTAP', 'DRI', 'SRE', 'DE', 'EXPE', 'PAYX', 'VLO', 'FCX', 'ROK', 'AVY', 
               'LUV', 'PH', 'SPG', 'DFS', 'CI', 'KO', 'PNC', 'CPB', 'WBA', 'MDLZ', 'TXN', 'A', 'EMN', 'HAS', 'STZ', 'D', 'ECL', 
               'IBM', 'QCOM', 'HUM', 'HSY', 'DIS', 'DTE', 'DOV', 'INTC', 'UPS', 'CLX', 'KR', 'PCAR', 'FE', 'BKR', 'LUMN', 'MO', 
               'APA', 'AMZN', 'NWL', 'ADM', 'BBY', 'NI', 'TJX', 'XEL', 'DVA', 'EL', 'EQR', 'EXPD', 'XRAY', 'DHI', 'CAH', 'MRK', 
               'MSFT', 'AON', 'BIIB', 'ICE', 'L', 'SNA', 'VZ', 'TT', 'NTRS', 'BXP', 'EBAY', 'BK', 'MAS', 'MCK', 'SLB', 'FTI', 'J', 
               'PPL', 'SYY', 'BAX', 'MCO', 'CME', 'ABT', 'ADP', 'TSN', 'MCD', 'ADBE', 'CVS']

start_time = "1/1/2008"
end_time = "12/29/2022"

fit_start_time = "1/1/2008"
fit_end_time = "12/31/2014"

valid_start_time = "1/1/2015"
valid_end_time = "12/31/2016"

test_start_time = "1/1/2017"
test_end_time = "12/29/2022"

provider_uri = "~/.qlib/qlib_data/my_data/sp500_components"
qlib.init(provider_uri=provider_uri)

dataset = create_dataset(
    instruments,
    start_time,
    end_time,
    fit_start_time,
    fit_end_time,
    valid_start_time,
    valid_end_time,
    test_start_time,
    test_end_time,
    handler_name="Alpha158",
    dataset_class="TSDatasetH"
)

# ---------- modelling ----------
EXP_NAME = "exp_gats_df"
model_conf = {
    "class": "GATs",
    "module_path": "qlib.contrib.model.pytorch_gats_ts_df",
    "kwargs": {
        "d_feat": 20,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.5, # 0.7
        "n_epochs": 1,
        "lr": 1e-4,
        "weight_decay": 0.001,
        "early_stop": 10,
        "metric": "loss",
        "loss": "mse",
        "base_model": "LSTM",
        "model_path": "/home/ashotnanyan/qlib/examples/benchmarks/LSTM/csi300_lstm_ts.pkl",
        "GPU": 0,
        "tensorboard_path": "/home/ashotnanyan/qlib/tensorboard_logs",
        "print_iter": 50
    },
}
model = init_instance_by_config(model_conf)

with R.start(experiment_name=EXP_NAME):

    model.fit(dataset)
    R.save_objects(trained_model=model)

    rec = R.get_recorder()
    rid = rec.id  # save the record id

    # Inference and saving signal
    sr = SignalRecord(model, dataset, rec)
    sr.generate()

# ---------- analysis ----------
with R.start(experiment_name=EXP_NAME):
    recorder = R.get_recorder(recorder_id="32a0a70dec2d4a6abe5d9e1a60a8f65f", experiment_name=EXP_NAME)
    model = recorder.load_object("trained_model")
    df = model.output_df(dataset)

trained_model_path = "/home/ashotnanyan/mlruns/1/32a0a70dec2d4a6abe5d9e1a60a8f65f/artifacts/trained_model"
model = R.load_object(trained_model_path)
df = model.output_df(dataset)

for instrument in instruments:
    d = df.xs(key=instrument, level=1)
    d_ = pd.read_csv(join("~/data_yf/", f"{instrument}.csv")).set_index("Date")
    d_.index = d_.index.astype('datetime64[ns]')
    
    d__ = pd.merge(d_, d, left_index=True, right_index=True)
    d__.index.name = "datetime"
    d__.to_csv(join("/home/ashotnanyan/gats_df/", f"{instrument}.csv"))

# ---------- dump all ----------
hidden_size = 64
base_feature_lbl = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
feature_lbl = [f"F_{i}" for i in range(1, hidden_size+1)]
base_feature_lbl.extend(feature_lbl)
base_feature_lbl.append("return")

dumping_engine = "~/qlib/scripts/dump_bin.py"
csv_path = "~/gats_df"
qlib_dir = "~/.qlib/qlib_data/my_data/gats_df"
date_field_name = "datetime"
include_fields = ",".join(base_feature_lbl)

# ---------- ALSTM with GATS DF ----------
import qlib
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord
from qlib.workflow.record_temp import SigAnaRecord
from qlib.workflow.record_temp import PortAnaRecord
from qlib.contrib.report import analysis_position
from qlib.utils import init_instance_by_config
from portfolio_management_2_stage.utils.creating_dataset import create_dataset

EXP_NAME = "alstm_with_gats_df"
instruments = ['PNW', 'MA', 'FDX', 'BEN', 'NKE', 'CSX', 'F', 'PKI', 'PHM', 'ADSK', 'JNJ', 'LMT', 'FIS', 'ITW', 'SHW', 'ADI', 
               'PRU', 'PFG', 'USB', 'GILD', 'LIN', 'FITB', 'UNM', 'NSC', 'HIG', 'AIZ', 'BDX', 'CMS', 'DVN', 'PEP', 'PGR', 'AMT', 
               'CMA', 'PWR', 'K', 'BKNG', 'ABC', 'CMI', 'COST', 'UNP', 'LEG', 'TROW', 'MMM', 'GD', 'CTSH', 'CMCSA', 'FISV', 
               'ZION', 'CL', 'HPQ', 'RF', 'CHRW', 'MAR', 'WEC', 'TPR', 'APH', 'LHX', 'CINF', 'PFE', 'JPM', 'SJM', 'HD', 'JNPR', 
               'IRM', 'RSG', 'LLY', 'KLAC', 'GIS', 'WMB', 'AIG', 'AIV', 'NRG', 'RL', 'GPC', 'XRX', 'TMO', 'UNH', 'CF', 'EOG', 
               'PEG', 'COF', 'STT', 'LH', 'GS', 'WMT', 'C', 'HST', 'OMC', 'PLD', 'PXD', 'ROST', 'SYK', 'OXY', 'DGX', 'AAPL', 
               'CTAS', 'GPS', 'IPG', 'GL', 'AXP', 'HON', 'KIM', 'ROP', 'AES', 'ES', 'PSA', 'AFL', 'KMB', 'CCL', 'ORLY', 'GWW', 
               'GE', 'EXC', 'MDT', 'SEE', 'TGT', 'TRV', 'MCHP', 'WDC', 'XOM', 'HBAN', 'APD', 'MKC', 'MS', 'HES', 'ETR', 'MMC', 
               'VNO', 'LNC', 'NEM', 'TAP', 'BAC', 'MRO', 'PEAK', 'IFF', 'AKAM', 'ALL', 'IP', 'VMC', 'FLS', 'DHR', 'MET', 'NUE', 
               'CNP', 'CAT', 'TXT', 'ED', 'CVX', 'SCHW', 'ETN', 'WM', 'WU', 'GOOG', 'MTB', 'SWK', 'RHI', 'KEY', 'AVB', 'GLW', 
               'CAG', 'AEE', 'T', 'PPG', 'ORCL', 'WELL', 'AMGN', 'SPGI', 'MU', 'CBRE', 'UAA', 'WY', 'EFX', 'IVZ', 'HAL', 'OKE', 
               'WFC', 'DUK', 'WYNN', 'WAT', 'PG', 'EA', 'NOC', 'BSX', 'NDAQ', 'LEN', 'BA', 'RTX', 'BMY', 'FMC', 'NVDA', 'WHR', 
               'YUM', 'NEE', 'AMP', 'SO', '^GSPC', 'EIX', 'CSCO', 'VFC', 'COP', 'EMR', 'ISRG', 'ZBH', 'AEP', 'AMAT', 'AZO', 'NOV', 
               'LOW', 'MSI', 'VRSN', 'TFC', 'INTU', 'SBUX', 'NTAP', 'DRI', 'SRE', 'DE', 'EXPE', 'PAYX', 'VLO', 'FCX', 'ROK', 'AVY', 
               'LUV', 'PH', 'SPG', 'DFS', 'CI', 'KO', 'PNC', 'CPB', 'WBA', 'MDLZ', 'TXN', 'A', 'EMN', 'HAS', 'STZ', 'D', 'ECL', 
               'IBM', 'QCOM', 'HUM', 'HSY', 'DIS', 'DTE', 'DOV', 'INTC', 'UPS', 'CLX', 'KR', 'PCAR', 'FE', 'BKR', 'LUMN', 'MO', 
               'APA', 'AMZN', 'NWL', 'ADM', 'BBY', 'NI', 'TJX', 'XEL', 'DVA', 'EL', 'EQR', 'EXPD', 'XRAY', 'DHI', 'CAH', 'MRK', 
               'MSFT', 'AON', 'BIIB', 'ICE', 'L', 'SNA', 'VZ', 'TT', 'NTRS', 'BXP', 'EBAY', 'BK', 'MAS', 'MCK', 'SLB', 'FTI', 'J', 
               'PPL', 'SYY', 'BAX', 'MCO', 'CME', 'ABT', 'ADP', 'TSN', 'MCD', 'ADBE', 'CVS']

col_list = ['F_1', 'F_2', 'F_3', 'F_4', 'F_5', 'F_6', 'F_7', 'F_8', 'F_9', 'F_10', 'F_11', 'F_12', 'F_13', 'F_14', 'F_15',
            'F_16', 'F_17', 'F_18', 'F_19', 'F_20', 'F_21', 'F_22', 'F_23', 'F_24', 'F_25', 'F_26', 'F_27', 'F_28', 'F_29', 'F_30',
            'F_31', 'F_32', 'F_33', 'F_34', 'F_35', 'F_36', 'F_37', 'F_38', 'F_39', 'F_40', 'F_41', 'F_42', 'F_43', 'F_44', 'F_45',
            'F_46', 'F_47', 'F_48', 'F_49', 'F_50', 'F_51', 'F_52', 'F_53', 'F_54', 'F_55', 'F_56', 'F_57', 'F_58', 'F_59', 'F_60',
            'F_61', 'F_62', 'F_63', 'F_64']

start_time = "1/1/2008"
end_time = "12/29/2022"

fit_start_time = "1/1/2008"
fit_end_time = "12/31/2014"

valid_start_time = "1/1/2015"
valid_end_time = "12/31/2016"

test_start_time = "1/1/2017"
test_end_time = "12/29/2022"

provider_uri = "~/.qlib/qlib_data/my_data/gats_df"
qlib.init(provider_uri=provider_uri)

dataset = create_dataset(
        instruments,
        start_time,
        end_time,
        fit_start_time,
        fit_end_time,
        valid_start_time,
        valid_end_time,
        test_start_time,
        test_end_time,
        handler_name="CustomFeaturesLabelALSTM",
        hl_module_path="portfolio_management_2_stage.data.data_loading",
        col_list=col_list,
        dataset_class="TSDatasetH" # DatasetH
)

model_conf = {
    "class": "ALSTM",
    "module_path": "qlib.contrib.model.pytorch_alstm_ts",
    "kwargs": {
        "d_feat": 64, # F_1 - F_64
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.8,
        "n_epochs": 200, # 20
        "lr": 1e-3,
        "early_stop": 10,
        # batch_size: 100 # 800
        "metric": "loss",
        "loss": "mse",
        "n_jobs": 20,
        "GPU": 0,
        "rnn_type": "GRU",
        "tensorboard_path": "/home/ashotnanyan/qlib/tensorboard_logs",
        "print_iter": 50
    },
}
model = init_instance_by_config(model_conf)

with R.start(experiment_name=EXP_NAME):

    model.fit(dataset)
    R.save_objects(trained_model=model)

    rec = R.get_recorder()
    rid = rec.id  # save the record id

    # Inference and saving signal
    sr = SignalRecord(model, dataset, rec)
    sr.generate()

# ---------- backtest and analysis ----------
BENCHMARK = "^GSPC"
test_start_time = "1/1/2017"
test_end_time = "12/28/2022" # one day before the end
rid = 'dac4e43398c04675a276dde3c75160d0'

port_analysis_config = {
    "executor": {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    },
    "strategy": {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {
            # "signal": ["<MODEL>", "<DATASET>"],
            "signal": "<PRED>",
            # "risk_degree": 1,
            "topk": 50,
            "n_drop": 5,
        },
    },
    "backtest": {
        "start_time": test_start_time,
        "end_time": test_end_time,
        "account": 100000000,
        "benchmark": BENCHMARK,
        "exchange_kwargs": {
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    },
}

with R.start(experiment_name=EXP_NAME, recorder_id=rid, resume=True):

    # signal-based analysis
    rec = R.get_recorder()
    sar = SigAnaRecord(rec)
    sar.generate()

    # portfolio-based analysis: backtest
    par = PortAnaRecord(rec, port_analysis_config, "day")
    par.generate()

# ---------- Analysis ----------
recorder = R.get_recorder(recorder_id=rid, experiment_name=EXP_NAME)

pred_df = recorder.load_object("pred.pkl")
report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

loaded_model = recorder.load_object("trained_model")

# the first set of graphs
analysis_position.report_graph(report_normal_df)
import numpy as np

idx = np.arange(0, 300)
u = np.tile(idx, 0)
j = np.arange(0, 600)
len(np.append(u, j))

k = 20
daily_index = np.arange(0, 528900, 300)
daily_count = np.repeat(300, len(a))
a = daily_index[0]
b = daily_count[0]
a_b = np.arange(a, b)
def iter():
    for idx, count in zip(daily_index, daily_count):
        k -= 1
        if k >= 0: 
            missing = np.tile(a_b, k)
            existing = np.arange(0, idx + count)
            yield np.append(missing, existing)
        else:
            yield np.arange(-k*idx, idx + count)
