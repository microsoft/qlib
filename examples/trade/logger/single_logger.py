import pandas as pd
import numpy as np
import os
from multiprocessing import Queue, Process
import time


def GLR(values):
    """

    Calculate -P(value | value > 0) / P(value | value < 0)

    """
    pos = []
    neg = []
    for i in values:
        if i > 0:
            pos.append(i)
        elif i < 0:
            neg.append(i)
    return -np.mean(pos) / np.mean(neg)


class DFLogger(object):
    """The logger for single-assert backtest.
    Would save .pkl and .log in log_dir


    """

    def __init__(self, log_dir, order_dir, writer=None):
        self.order_dir = order_dir + "/"
        self.log_dir = log_dir + "/"
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.queue = Queue(100000)
        self.raw_log_dir = self.log_dir

    @staticmethod
    def _worker(log_dir, order_dir, queue):
        df_cache = {}
        stat_cache = {}
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        while True:
            info = queue.get(block=True)
            if info == "stop":
                summary = {}
                for k, v in stat_cache.items():
                    if not k.startswith("money"):
                        summary[k + "_std"] = np.nanstd(v)
                        summary[k + "_mean"] = np.nanmean(v)
                try:
                    for k in ["PR_sell", "ffr_sell", "PA_sell"]:
                        summary["weighted_" + k] = np.average(stat_cache[k], weights=stat_cache["money_sell"])
                except:
                    # summary["weighted_" + k] = np.average(stat_cache[k], weights=stat_cache['money_sell'])
                    pass
                try:
                    for k in ["PR_buy", "ffr_buy", "PA_buy"]:
                        summary["weighted_" + k] = np.average(stat_cache[k], weights=stat_cache["money_buy"])
                except:
                    pass
                try:
                    for k in ["obs0_PR", "ffr", "PA"]:
                        summary["weighted_" + k] = np.average(stat_cache[k], weights=stat_cache["money"])
                except:
                    pass
                summary["GLR"] = GLR(stat_cache["PA"])
                try:
                    summary["GLR_sell"] = GLR(stat_cache["PA_sell"])
                except:
                    pass
                try:
                    summary["GLR_buy"] = GLR(stat_cache["PA_buy"])
                except:
                    pass
                queue.put(summary)
                break
            elif len(info) == 0:
                continue
            else:
                df = info.pop("df")
                res = info.pop("res")
                ins = df.index[0][0]
                if ins not in df_cache:
                    df_cache[ins] = (
                        [],
                        [],
                        (pd.read_pickle(order_dir + ins + ".pkl.target")['amount'] != 0).sum(),
                    )
                df_cache[ins][0].append(df)
                df_cache[ins][1].append(res)
                if len(df_cache[ins][0]) == df_cache[ins][2]:
                    pd.concat(df_cache[ins][0]).to_pickle(log_dir + ins + ".log")
                    pd.concat(df_cache[ins][1]).to_pickle(log_dir + ins + ".pkl")
                    del df_cache[ins]
                for k, v in info.items():
                    if k not in stat_cache:
                        stat_cache[k] = []
                    if hasattr(v, "__len__"):
                        stat_cache[k] += list(v)
                    else:
                        stat_cache[k].append(v)

    def reset(self):
        """ """
        while not self.queue.empty():
            self.queue.get()
        assert self.queue.empty()
        self.child = Process(target=self._worker, args=(self.log_dir, self.order_dir, self.queue), daemon=True,)
        self.child.start()

    def set_step(self, step):

        self.log_dir = f"{self.raw_log_dir}{step}/"
        self.reset()

    def __call__(self, infos):
        for info in infos:
            if "env_id" in info:
                info.pop("env_id")
        self.update(infos)

    def update(self, infos):
        """store values in info into the logger"""
        for info in infos:
            self.queue.put(info, block=True)

    def summary(self):
        """:return: The mean and std of values in infos stored in logger"""
        summary = {}
        self.queue.put("stop", block=True)
        self.child.join()
        self.child.close()
        assert self.queue.qsize() == 1
        summary = self.queue.get()

        return summary


class InfoLogger(DFLogger):
    """ """

    def __init__(self, *args):
        self.stat_cache = {}
        self.queue = Queue(10000)
        self.child = Process(target=self._worker, args=(self.queue,), daemon=True)
        self.child.start()

    def _worker(logdir, queue):
        stat_cache = {}
        while True:
            info = queue.get(block=True)
            if info == "stop":
                summary = {}
                for k, v in stat_cache.items():
                    if not k.startswith("money"):
                        summary[k + "_std"] = np.nanstd(v)
                        summary[k + "_mean"] = np.nanmean(v)
                try:
                    for k in ["PR_sell", "ffr_sell", "PA_sell"]:
                        summary["weighted_" + k] = np.average(stat_cache[k], weights=stat_cache["money_sell"])
                except:
                    pass
                try:
                    for k in ["PR_buy", "ffr_buy", "PA_buy"]:
                        summary["weighted_" + k] = np.average(stat_cache[k], weights=stat_cache["money_buy"])
                except:
                    pass
                try:
                    for k in ["obs0_PR", "ffr", "PA"]:
                        summary["weighted_" + k] = np.average(stat_cache[k], weights=stat_cache["money"])
                except:
                    pass
                summary["GLR"] = GLR(stat_cache["PA"])
                try:
                    summary["GLR_sell"] = GLR(stat_cache["PA_sell"])
                except:
                    pass
                try:
                    summary["GLR_buy"] = GLR(stat_cache["PA_buy"])
                except:
                    pass
                queue.put(summary)
                stat_cache = {}
                time.sleep(5)
                continue
            if len(info) == 0:
                continue
            for k, v in info.items():
                if k == "res" or k == "df":
                    continue
                if k not in stat_cache:
                    stat_cache[k] = []
                if hasattr(v, "__len__"):
                    stat_cache[k] += list(v)
                else:
                    stat_cache[k].append(v)

    def _update(self, info):
        if len(info) == 0:
            return
        ins = df.index[0][0]
        for k, v in info.items():
            if k not in self.stat_cache:
                self.stat_cache[k] = []
            if hasattr(v, "__len__"):
                self.stat_cache[k] += list(v)
            else:
                self.stat_cache[k].append(v)

    def summary(self):
        """ """
        while not self.queue.empty():
            # print('not empty')
            # print(self.queue.qsize())
            time.sleep(1)
        self.queue.put("stop")
        # self.child.join()
        time.sleep(1)
        while not self.queue.qsize() == 1:
            # print(self.queue.qsize())
            time.sleep(1)
        assert self.queue.qsize() == 1
        summary = self.queue.get()

        return summary

    def set_step(self, step):
        return
