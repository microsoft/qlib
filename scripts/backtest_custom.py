import numpy as np
import qlib
from qlib.utils.time import Freq
from qlib.backtest import backtest, executor
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

# init qlib
qlib.init(provider_uri='~/.qlib/qlib_data/my_data/sp500_components')

BENCH = "^GSPC"
# Benchmark is for calculating the excess return of your strategy.
# Its data format will be like **ONE normal instrument**.
# For example, you can query its data with the code below
# `D.features(["SH000300"], ["$close"], start_time='2010-01-01', end_time='2017-12-31', freq='day')`
# It is different from the argument `market`, which indicates a universe of stocks (e.g. **A SET** of stocks like csi300)
# For example, you can query all data from a stock market with the code below.
# ` D.features(D.instruments(market='csi300'), ["$close"], start_time='2010-01-01', end_time='2017-12-31', freq='day')`

pred_score = np.load("/home/ashotnanyan/qlib/examples/test_gats/decay_001/1/69bdd9a5a84c48e3a1852e76809315e1/artifacts/pred.pkl", allow_pickle=True)

FREQ = "day"
STRATEGY_CONFIG = {
    "topk": 50,
    "n_drop": 5,
    # "riskmodel_path": "/home/erohar/qlib/examples/portfolio/riskdata",
    # pred_score, pd.Series
    "signal": pred_score,
}

EXECUTOR_CONFIG = {
    "time_per_step": "day",
    "generate_portfolio_metrics": True,
}

backtest_config = {
    "start_time": "2017-01-01",
    "end_time": "2022-12-29",
    "account": 100000000,
    "benchmark": BENCH,
    "exchange_kwargs": {
        "codes": ['PNW', 'MA', 'FDX', 'BEN', 'NKE', 'CSX', 'F', 'PKI', 'PHM', 'ADSK', 'JNJ', 'LMT', 'FIS', 'ITW', 'SHW', 'ADI', 
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
                'YUM', 'NEE', 'AMP', 'SO', 'EIX', 'CSCO', 'VFC', 'COP', 'EMR', 'ISRG', 'ZBH', 'AEP', 'AMAT', 'AZO', 'NOV', 
                'LOW', 'MSI', 'VRSN', 'TFC', 'INTU', 'SBUX', 'NTAP', 'DRI', 'SRE', 'DE', 'EXPE', 'PAYX', 'VLO', 'FCX', 'ROK', 'AVY', 
                'LUV', 'PH', 'SPG', 'DFS', 'CI', 'KO', 'PNC', 'CPB', 'WBA', 'MDLZ', 'TXN', 'A', 'EMN', 'HAS', 'STZ', 'D', 'ECL', 
                'IBM', 'QCOM', 'HUM', 'HSY', 'DIS', 'DTE', 'DOV', 'INTC', 'UPS', 'CLX', 'KR', 'PCAR', 'FE', 'BKR', 'LUMN', 'MO', 
                'APA', 'AMZN', 'NWL', 'ADM', 'BBY', 'NI', 'TJX', 'XEL', 'DVA', 'EL', 'EQR', 'EXPD', 'XRAY', 'DHI', 'CAH', 'MRK', 
                'MSFT', 'AON', 'BIIB', 'ICE', 'L', 'SNA', 'VZ', 'TT', 'NTRS', 'BXP', 'EBAY', 'BK', 'MAS', 'MCK', 'SLB', 'FTI', 'J', 
                'PPL', 'SYY', 'BAX', 'MCO', 'CME', 'ABT', 'ADP', 'TSN', 'MCD', 'ADBE', 'CVS'],
        "freq": FREQ,
        "limit_threshold": 0.095,
        "deal_price": "close",
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5,
    },
}

# strategy object
strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
# executor object
executor_obj = executor.SimulatorExecutor(**EXECUTOR_CONFIG)
# backtest
portfolio_metric_dict, indicator_dict = backtest(executor=executor_obj, strategy=strategy_obj, **backtest_config)
analysis_freq = "{0}{1}".format(*Freq.parse(FREQ))
# backtest info
report_normal, positions_normal = portfolio_metric_dict.get(analysis_freq)

report_normal.to_pickle('~/gats_report_normal_gspc_in_train.pkl')
print(report_normal)