import numpy as np
import pandas as pd
import os
import time
import datetime
from joblib import Parallel, delayed

data_path = '../data/'
in_dir = os.path.join(data_path, 'backtest/')

### create order folders ####

def generate_order(df, start, end):
#     df['date'] = df.index.map(lambda x: x[1].date())
#     df.set_index('date', append=True, inplace=True)
    df = df.groupby('date').take(range(start, end)).droplevel(level=0)
    div = df['$volume0'].rolling((end - start)*60).mean().shift(1).groupby(level='date').transform('first')
    order = df.groupby(level=(2, 0)).mean().dropna()
    order = pd.DataFrame(order)
    order['amount'] = np.random.lognormal(-3.28, 1.14) * order['$volume0']
    order['order_type'] = 0
    order = order.drop(columns=["$volume0", "$vwap0"])
    return order

def w_order(f, start, end):
    df = pd.read_pickle(in_dir + f)
    #df['date'] = df.index.get_level_values(1).map(lambda x: x.date())
    #df = df.set_index('date', append=True, drop=True)
#     old_order = pd.read_pickle('../v-zeh/full-07-20/order/ratio_test/' + f)
    order = generate_order(df, start, end)
#     order = order[order.index.isin(old_order.index)]
    order_train = order[order.index.get_level_values(0) < '2020-12-01']
    order_test = order[order.index.get_level_values(0) >= '2020-12-01']
    order_valid = order_test[order_test.index.get_level_values(0) < '2021-01-01']
    order_test = order_test[order_test.index.get_level_values(0) >= '2021-01-01']
    if len(order_train) > 0:
        train_path = os.path.join(data_path, "order/train/")
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        order_train.to_pickle(train_path + f[:-9] + '.target')
    if len(order_valid) > 0:
        valid_path = os.path.join(data_path, "order/valid/")
        if not os.path.exists(valid_path):
            os.makedirs(valid_path)
        order_valid.to_pickle(valid_path + f[:-9] + '.target')
    if len(order_test) > 0:
        test_path = os.path.join(data_path, "order/test/")
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        order_test.to_pickle(test_path + f[:-9] + '.target')
    if len(order) > 0:
        all_path = os.path.join(data_path, "order/all/")
        if not os.path.exists(all_path):
            os.makedirs(all_path)
        order_test.to_pickle(all_path + f[:-9] + '.target')
    return 0

res = Parallel(n_jobs=64)(delayed(w_order)(f, 0, 239) for f in os.listdir(in_dir))
print(sum(res))
