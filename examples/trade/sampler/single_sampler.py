import pandas as pd
import numpy as np
from multiprocessing.context import Process
from multiprocessing import Queue

import os
import sys

sys.path.append("..")


def toArray(data):
    if type(data) == np.ndarray:
        return data

    elif type(data) == list:
        data = np.array(data)
        return data

    elif type(data) == pd.DataFrame:
        share_index = toArray(data.index)
        share_value = toArray(data.values)
        share_colmns = toArray(data.columns)
        return share_index, share_value, share_colmns

    else:
        try:
            share_array = np.array(data)
            return share_array
        except:
            raise NotImplementedError


class Sampler:
    """The sampler for training of single-assert RL."""

    def __init__(self, config):
        self.raw_dir = config["raw_dir"] + "/"
        self.order_dir = config["order_dir"] + "/"
        self.ins_list = [f[:-11] for f in os.listdir(self.order_dir) if f.endswith("target")]
        self.features = config["features"]
        self.queue = Queue(1000)
        self.child = None
        self.ins = None
        self.raw_df = None
        self.df_list = None
        self.order_df = None

    @staticmethod
    def _worker(order_dir, raw_dir, features, ins_list, queue):
        ins = None
        index = 0
        date_list = []
        while True:
            if ins is None or index == len(date_list):
                ins = np.random.choice(ins_list, 1)[0]
                # print(ins)
                order_df = pd.read_pickle(order_dir + ins + ".pkl.target")
                feature_df_list = []
                for feature in features:
                    feature_df_list.append(pd.read_pickle(f"{feature['loc']}/{ins}.pkl"))
                raw_df = pd.read_pickle(raw_dir + ins + ".pkl.backtest")
                date_list = order_df.index.get_level_values(0).tolist()
                index = 0
            date = date_list[index]
            day_order_df = order_df.iloc[index]
            target = day_order_df["amount"]
            index += 1
            if target == 0:
                continue
            day_feature_dfs = []
            day_raw_df = raw_df.loc[pd.IndexSlice[ins, :, date]]
            is_buy = bool(day_order_df["order_type"])
            for df in feature_df_list:
                day_feature_dfs.append(df.loc[ins, date].values)
            day_feature_dfs = np.array(day_feature_dfs)
            day_raw_df_index, day_raw_df_value, day_raw_df_column = toArray(day_raw_df)
            day_feature_dfs_ = toArray(day_feature_dfs)
            queue.put(
                (ins, date, day_raw_df_value, day_raw_df_column, day_raw_df_index, day_feature_dfs_, target, is_buy,),
                block=True,
            )

    def _sample_ins(self):
        """ """
        return np.random.choice(self.ins_list, 1)[0]

    def reset(self):
        """ """
        if self.child is None:
            self.child = Process(
                target=self._worker,
                args=(self.order_dir, self.raw_dir, self.features, self.ins_list, self.queue,),
                daemon=True,
            )
            self.child.start()

    def sample(self):
        """ """
        sample = self.queue.get(block=True)
        return sample

    def stop(self):
        """ """
        try:
            self.child.terminate()
        except:
            for p in self.child:
                p.terminate()


class TestSampler(Sampler):
    """The sampler for backtest of single-assert strategies."""

    def __init__(self, config):
        super().__init__(config)
        self.ins_index = -1

    def _sample_ins(self):
        """ """
        self.ins_index += 1
        if self.ins_index >= len(self.ins_list):
            return None
        else:
            return self.ins_list[self.ins_index]

    @staticmethod
    def _worker(order_dir, raw_dir, features, ins_list, queue):
        for ins in ins_list:
            order_df = pd.read_pickle(order_dir + ins + ".pkl.target")
            df_list = []
            for feature in features:
                df_list.append(pd.read_pickle(f"{feature['loc']}/{ins}.pkl"))
            raw_df = pd.read_pickle(raw_dir + ins + ".pkl.backtest")
            date_list = order_df.index.get_level_values(0).tolist()
            for index in range(len(date_list)):
                date = date_list[index]
                day_df_list = []
                day_raw_df = raw_df.loc[pd.IndexSlice[ins, :, date]]
                day_order_df = order_df.iloc[index]
                target = day_order_df["amount"]
                if target == 0:
                    continue
                is_buy = bool(day_order_df["order_type"])
                for df in df_list:
                    day_df_list.append(df.loc[ins, date].values)
                day_feature_dfs = np.array(day_df_list)
                day_raw_df_index, day_raw_df_value, day_raw_df_column = toArray(day_raw_df)
                day_feature_dfs_ = toArray(day_feature_dfs)
                queue.put(
                    (
                        ins,
                        date,
                        day_raw_df_value,
                        day_raw_df_column,
                        day_raw_df_index,
                        day_feature_dfs_,
                        target,
                        is_buy,
                    ),
                    block=True,
                )
        for _ in range(100):
            queue.put(None)

    def reset(self, order_dir=None):
        """

        reset the sampler and change self.order_dir if order_dir is not None.

        """
        if order_dir:
            self.order_dir = order_dir
            self.ins_list = [f[:-11] for f in os.listdir(self.order_dir) if f.endswith("target")]
        if not self.child is None:
            self.child.terminate()
            while not self.queue.empty():
                self.queue.get()
        self.child = Process(
            target=self._worker,
            args=(self.order_dir, self.raw_dir, self.features, self.ins_list, self.queue,),
            daemon=True,
        )
        self.child.start()
