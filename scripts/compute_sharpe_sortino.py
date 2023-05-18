########## some functions for computing Sharpe and Sortino ##########

import pandas as pd
import numpy as np

# developed by Bilor
def Sharpe_and_Sortino(returns, target_return=0.1,  risk_free=0.001):
    L=len(returns)
    risk_adjusted = risk_free/L
    target_adjusted = target_return/L
    eps=1e-6
    
    SP_sharpe = (returns.mean() - risk_adjusted)  / (np.std(returns)+eps)*np.sqrt(L)

    cum_return = np.prod(returns[1:]+1)-1
    SP_sortino_num = cum_return - risk_adjusted 
    downside_diff =  returns - target_adjusted
    downside_diff = downside_diff[downside_diff < 0]
    if np.any(downside_diff):
        downside_diff = np.power(downside_diff, 2)
        downside_deviation = np.sqrt(downside_diff.mean())
    else:
        downside_deviation = 0
    SP_sortino = SP_sortino_num / (downside_deviation + eps)

    return SP_sharpe, SP_sortino

# developed by Ashot from code provided by Bilor
def cum_returns(input_data, win_size, start, end):
    rets_for_sharpe = []
    idxs = input_data.index
    i_start = idxs.get_loc(start)
    i_end = idxs.get_loc(end)

    data = input_data.iloc[i_start:i_end]
    for i in range(win_size, data.shape[0], win_size):
        item_1 = data.iloc[i-win_size]["Prices"]
        item_2 = data.iloc[i]["Prices"]
        cum_ret = item_2/item_1-1
        rets_for_sharpe.append(cum_ret)

    return rets_for_sharpe

# developed by Bilor
def granular_frame_extractor(input_data, win_size, target_return, risk_free, interval = '1y', start = "2000-01-10", end = "2022-01-14"):
    dates = []; timesteps = []; Sharpe =[]; Sortino = []; rets_for_sharpe = []
    years = np.fromstring(interval[:-1], dtype=np.int32, sep=' ')
    k = np.floor(years*252/win_size)*win_size  #k=250
    idxs = input_data.index
    i_start = idxs.get_loc(start)
    i_end = idxs.get_loc(end)

    data = input_data.iloc[i_start:i_end]
    #print(data.shape)

    for i in range(win_size, data.shape[0], win_size):
        item_1 = data.iloc[i-win_size]["Prices"]
        item_2 = data.iloc[i]["Prices"]
        cum_ret = item_2/item_1-1
        rets_for_sharpe.append(cum_ret)
        
        #print("i=", i)
        if i % k == 0:
            
            SP_sharpe, SP_sortino = Sharpe_and_Sortino(np.array(rets_for_sharpe),
                                                       target_return=target_return,
                                                       risk_free=risk_free)
            Sharpe.append(SP_sharpe)
            Sortino.append(SP_sortino)
            rets_for_sharpe = []
    
            data_i = data.iloc[i:i+1].index
            dates.append(data_i)
            timesteps.append(i) 
        #if i>1050:
        #   break
    timesteps = np.array(timesteps); Sharpe = np.array(Sharpe); Sortino = np.array(Sortino)
    timesteps = timesteps.reshape(-1, 1)
    Sharpe = Sharpe.reshape(-1, 1); Sortino = Sortino.reshape(-1, 1)
    dates = np.array(dates)
    a = np.hstack((timesteps, Sharpe, Sortino))
    # print(np.shape(a))
    df = pd.DataFrame(a, index = dates, columns = ["timesteps", "Sharpe", "Sortino"])
    return df

# developed by Ashot
def create_ss(accounts={}, 
              win_size=20, 
              win_size_overall=240,
              target_return=0.1,
              risk_free=0.001,
              interval="1y",
              interval_overall="6y",
              start="2017-01-03",
              end="2022-12-28"):
    ss_lst = []
    
    # 6-year target return and risk-free rate
    target_return_overall = int(interval_overall[0]) * target_return
    risk_free_overall = int(interval_overall[0]) * risk_free
    
    for i, account in enumerate(accounts.values()):
        ss_account = granular_frame_extractor(account,
                                              win_size=win_size,
                                              target_return=target_return,
                                              risk_free=risk_free,
                                              interval=interval,
                                              start=start, 
                                              end=end)
        ss_account_overall = granular_frame_extractor(account,
                                                      win_size=win_size_overall,
                                                      target_return=target_return_overall,
                                                      risk_free=risk_free_overall,
                                                      interval=interval_overall,
                                                      start=start, 
                                                      end=end)
        ret = cum_returns(account,
                          win_size=win_size_overall,
                          start=start,
                          end=end)
        
        ss_account["Returns"] = ret
        ss_account_overall["Returns"] = np.prod(np.array(ret) + 1) - 1
        ss_account_overall.index = ["overall"]

        ss_account = pd.concat([ss_account, ss_account_overall], axis=0)
        if i == 0:
            ss_timesteps = ss_account["timesteps"]
            ss_lst.append(ss_timesteps)
        ss_account = ss_account[["Sharpe", "Sortino", "Returns"]]
        ss_lst.append(ss_account)
    
    keys = ["timesteps"]+list(accounts.keys())
    ss = pd.concat(ss_lst, axis=1, keys=keys)
    ss = ss.reorder_levels([1, 0], axis=1)
    ss.sort_index(axis=1, level=0, inplace=True)

    cols_0 = ["timesteps", "Returns", "Sharpe", "Sortino"]
    new_cols_0 = ss.columns.reindex(cols_0, level=0)
    ss = ss.reindex(columns=new_cols_0[0])

    cols_1 = list(accounts.keys())
    new_cols_1 = ss.iloc[:, 1:].columns.reindex(cols_1, level=1)
    ss = pd.concat([ss.iloc[:, 0], 
                     ss.iloc[:, 1:].reindex(columns=new_cols_1[0])], axis=1)
    
    return ss

def prepare_data(report_normal, mode="strategy"):
    if mode=="strategy":
        account = report_normal["account"].to_frame(name="Prices")
        account.index = account.index.map(str)
        account.index = account.index.map(lambda x: x[:10])
        return account
    report_normal["benchmark1"] = report_normal["bench"] + 1
    report_normal["benchmark2"] = report_normal["benchmark1"].cumprod()
    account_bench = 100_000_000 * report_normal["benchmark2"].to_frame(name="Prices")
    account_bench.index = account_bench.index.map(str)
    account_bench.index = account_bench.index.map(lambda x: x[:10])
    return account_bench


########## invoking the function `create_ss` ##########
import qlib
from qlib.workflow import R

provider_uri = "~/.qlib/qlib_data/my_data/sp500_components"
qlib.init(provider_uri=provider_uri)

gats_gspc_in_train = R.load_object("/home/ashotnanyan/gats_report_normal_gspc_in_train.pkl")
alstm_with_gats = R.load_object("/home/ashotnanyan/mlruns/3/dac4e43398c04675a276dde3c75160d0/artifacts/portfolio_analysis/report_normal_1day.pkl")
gats_lamb08_gspc_in_train = R.load_object("/home/ashotnanyan/qlib/examples/test_precise_margin_ranking/lamb_08_btest/1/7664c61f96554a7482f6e64a018a2233/artifacts/portfolio_analysis/report_normal_1day.pkl")
gats_alstm_lambda_in_one_loss = R.load_object("/home/ashotnanyan/qlib/examples/test_gats_alstm/08_in_one_loss/1/4a0bfff99d7842dfbaacbd8bce5c2c23/artifacts/portfolio_analysis/report_normal_1day.pkl")
gats_lamb08_sector_feat_etf = R.load_object("/home/ashotnanyan/qlib/examples/test_precise_margin_ranking/lamb_08_w_etf_feat_inst/1/bbedbb9918cd4724ad2d703687887b68/artifacts/portfolio_analysis/report_normal_1day.pkl")

account_gats_gspc_in_train = prepare_data(gats_gspc_in_train)
account_alstm_with_gats = prepare_data(alstm_with_gats)
account_gats_lamb08_gspc_in_train = prepare_data(gats_lamb08_gspc_in_train)
account_gats_alstm_lambda_in_one_loss = prepare_data(gats_alstm_lambda_in_one_loss)
account_gats_lamb08_sector_feat_etf = prepare_data(gats_lamb08_sector_feat_etf)
account_bench = prepare_data(gats_gspc_in_train, mode="benchmark")

accounts = {"BENCH": account_bench, 
            "GATs": account_gats_gspc_in_train,
            "ALSTM_with_GATs": account_alstm_with_gats, 
            "GATs_lamb08": account_gats_lamb08_gspc_in_train, 
            "GATs_ALSTM_lambda_in_one_loss": account_gats_alstm_lambda_in_one_loss,
            "GATs_lamb08_sector_feat_etf": account_gats_lamb08_sector_feat_etf}

print(create_ss(accounts=accounts))