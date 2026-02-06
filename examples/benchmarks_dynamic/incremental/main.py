# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import copy
import time
import traceback
from pathlib import Path
from pprint import pprint
from typing import Optional

import sys
import pandas as pd
import fire

DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.append(str(DIRNAME))
sys.path.append(str(DIRNAME.parent.parent.parent))

import qlib
from qlib.workflow.task.utils import TimeAdjuster
from qlib.utils.data import deepcopy_basic_type
from qlib.workflow.record_temp import SigAnaRecord, PortAnaRecord
from qlib.data.dataset import Dataset
from qlib import auto_init
from qlib.utils import init_instance_by_config
from qlib.workflow import R, Experiment
from qlib.tests.data import GetData

from qlib.data.dataset.handler import DataHandlerLP

from qlib.contrib.meta.incremental.model import MetaModelInc, DoubleAdaptManager
from qlib.contrib.meta.incremental.dataset import MetaDatasetInc
from qlib.contrib.meta.incremental.utils import *
from examples.benchmarks_dynamic.incremental.benchmark import Benchmark


class Incremental:
    """
    Example:
    python -u main.py run_all --forecast_model GRU -num_head 8 --tau 10 --first_order True --adapt_x True --adapt_y True --market csi300 --data_dir crowd_data --rank_label False
    """

    def __init__(
            self,
            data_dir="cn_data",
            root_path='~/.qlib/qlib_data/',
            market="csi300",
            horizon=1,
            alpha=360,
            step=20,
            rank_label=False,
            forecast_model="GRU",
            lr=0.01,
            lr_model=0.001,
            reg=0.5,
            num_head=8,
            tau=10,
            first_order=True,
            adapt_x=True,
            adapt_y=True,
            naive=False,
            save=False,
            begin_valid_epoch=20,
            preprocess_tensor=True,
            use_extra=False,
            reload_tag: Optional[str] = None,
            tag="",
            h_path=None,
            test_start=None,
            test_end=None,
    ):
        """
        :param data_dir: str
            source data dictionary under root_path
        :param root_path: str
            the root path of source data. '~/.qlib/qlib_data/' by default.
        :param market: str
            'csi300' or 'csi500'
        :param horizon: int
            define the stock price trend
        :param alpha: int
            360 or 158
        :param step: int
            incremental task interval, i.e., timespan of incremental data or test data
        :param rank_label: boolean
            If True, use CSRankNorm for label preprocessing; Otherwise, use CSZscoreNorm
        :param forecast_model: str
            consistent with directory name under examples/benchmarks
        :param lr: float
            learning rate of data adapter
        :param lr_model: float
            learning rate of forecast model and model adapter
        :param reg: float
            regularization strength
        :param num_head:
            number of transformation heads
        :param tau:
            softmax temperature
        :param first_order:
            whether use first-order approximation version of MAML
        :param adapt_x:
            whether adapt features
        :param adapt_y:
            whether adapt labels
        :param naive:
            if True, degrade to naive incremental baseline
        :param begin_valid_epoch:
            accelerate offline training by leaving out some valid epochs
        :param save:
            whether to save the checkpoints
        :param reload_tag (list):
            if None, train from scratch; otherwise, reload checkpoints from the previous experiments
        :param preprocess_tensor:
            If False, transform each batch from `numpy.ndarray` to `torch.Tensor` (slow, not recommended)
        :param use_extra:
            If True, use extra segments for upper-level optimization (not recommended when step is large enough)
        :param tag: str
            distinguish experiment name
        :param reload_tag:
            experiment name to reload
        :param h_path:
            prefetched handler file path to load
        :param test_start:
            the start date of test data
        :param test_end:
            the end date of test data
        """
        self.reload_tag = reload_tag
        self.save = save
        self.data_dir = data_dir
        self.market = market
        if self.data_dir == "us_data":
            if self.market == "sp500":
                self.benchmark = "^gspc"
            else:
                self.benchmark = "^ndx"
        elif self.market == "csi500":
            self.benchmark = "SH000905"
        elif self.market == "csi100":
            self.benchmark = "SH000903"
        else:
            self.benchmark = "SH000300"
        if data_dir == "cn_data":
            GetData().qlib_data(target_dir=root_path + "cn_data", exists_skip=True)
            auto_init()
        else:
            qlib.init(
                provider_uri=root_path + data_dir, region="us" if self.data_dir == "us_data" else "cn",
            )
        self.step = step
        self.horizon = horizon
        self.forecast_model = forecast_model  # downstream forecasting models' type
        self.alpha = alpha
        self.tag = tag
        self.rank_label = rank_label
        self.lr = lr
        self.lr_model = lr_model
        self.num_head = num_head
        self.temperature = tau
        self.first_order = first_order
        self.naive = naive
        self.adapt_x = adapt_x
        self.adapt_y = adapt_y
        self.reg = reg
        self.is_rnn = self.forecast_model in ["GRU", "LSTM", "ALSTM"]
        self.not_sequence = self.forecast_model in ["MLP", 'Linear'] and self.alpha == 158
        self.h_path = h_path
        self.basic_task = Benchmark(
            data_dir=self.data_dir,
            market=self.market,
            model_type=self.forecast_model,
            horizon=self.horizon,
            alpha=self.alpha,
            rank_label=self.rank_label,
            lr=lr_model,
            early_stop=8,
            init_data=False,
            h_path=h_path,
            test_start=test_start,
            test_end=test_end,
        ).basic_task()
        self.begin_valid_epoch = begin_valid_epoch
        self.preprocess_tensor = preprocess_tensor
        self.use_extra = use_extra
        R.set_uri((DIRNAME / 'mlruns').as_uri())

    @property
    def meta_exp_name(self):
        return f"{self.market}_{self.forecast_model}_alpha{self.alpha}_horizon{self.horizon}_step{self.step}_rank{self.rank_label}_{self.tag}"

    def dump_data(self):
        segments = self.basic_task["dataset"]["kwargs"]["segments"]
        t = deepcopy_basic_type(self.basic_task)
        t["dataset"]["kwargs"]["segments"]["train"] = (
            segments["train"][0],
            segments["test"][1],
        )
        ds = init_instance_by_config(t["dataset"], accept_types=Dataset)
        data = ds.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if t["dataset"]["class"] == "TSDatasetH":
            data.config(fillna_type="ffill+bfill")  # process nan brought by dataloader

        ta = TimeAdjuster(future=True, end_time=segments['test'][1])
        assert ta.align_seg(t["dataset"]["kwargs"]["segments"]["train"])[0] == data.index[0][0]
        # assert ta.align_seg(t["dataset"]["kwargs"]["segments"]["train"])[1] == data.index[-1][0]

        rolling_task = deepcopy_basic_type(self.basic_task)
        if "pt_model_kwargs" in rolling_task["model"]["kwargs"] and self.alpha == 158:
            self.factor_num = rolling_task["model"]["kwargs"]["pt_model_kwargs"]["input_dim"]
        elif "d_feat" in rolling_task["model"]["kwargs"]:
            self.factor_num = rolling_task["model"]["kwargs"]["d_feat"]
        else:
            self.factor_num = 6 if self.alpha == 360 else 20

        trunc_days = self.horizon if self.data_dir == "us_data" else (self.horizon + 1)
        segments = rolling_task["dataset"]["kwargs"]["segments"]
        train_begin = segments["train"][0]
        train_end = ta.get(ta.align_idx(train_begin) + self.step - 1)
        test_begin = ta.get(ta.align_idx(train_begin) + self.step - 1 + trunc_days)
        test_end = rolling_task["dataset"]["kwargs"]["segments"]["valid"][1]
        extra_begin = ta.get(ta.align_idx(train_end) + 1)
        extra_end = ta.get(ta.align_idx(test_begin) - 1)
        test_end = ta.get(ta.align_idx(test_end) - self.step)
        seperate_point = str(rolling_task["dataset"]["kwargs"]["segments"]["valid"][0])
        rolling_task["dataset"]["kwargs"]["segments"] = {
            "train": (train_begin, train_end),
            "test": (test_begin, test_end),
        }
        if self.use_extra:
            rolling_task["dataset"]["kwargs"]["segments"]["extra"] = (extra_begin, extra_end)

        kwargs = dict(
            task_tpl=rolling_task,
            step=self.step,
            segments=seperate_point,
            task_mode="train",
        )
        if self.forecast_model == "MLP" and self.alpha == 158:
            kwargs.update(task_mode="test")
        md_offline = MetaDatasetInc(data=data, **kwargs)
        md_offline.meta_task_l = preprocess(
            md_offline.meta_task_l,
            factor_num=self.factor_num,
            is_mlp=self.forecast_model == "MLP",
            alpha=self.alpha,
            step=self.step,
            H=self.horizon if self.data_dir == "us_data" else (1 + self.horizon),
            not_sequence=self.not_sequence,
            to_tensor=self.preprocess_tensor
        )

        L = md_offline.meta_task_l[0].get_meta_input()["X_test"].shape[1]
        if self.not_sequence:
            self.x_dim = L
            self.factor_num = self.x_dim
        else:
            self.x_dim = self.factor_num * L

        train_begin = segments["valid"][0]
        train_end = ta.get(ta.align_idx(train_begin) + self.step - 1)
        test_begin = ta.get(ta.align_idx(train_begin) + self.step - 1 + trunc_days)
        extra_begin = ta.get(ta.align_idx(train_end) + 1)
        extra_end = ta.get(ta.align_idx(test_begin) - 1)
        rolling_task["dataset"]["kwargs"]["segments"] = {
            "train": (train_begin, train_end),
            "test": (test_begin, segments["test"][1]),
        }
        if self.use_extra:
            rolling_task["dataset"]["kwargs"]["segments"]["extra"] = (extra_begin, extra_end)

        kwargs.update(task_tpl=rolling_task, segments=0.0)
        if self.forecast_model == "MLP" and self.alpha == 158:
            kwargs.update(task_mode="test")
            data_I = ds.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        else:
            data_I = None
        md_online = MetaDatasetInc(data=data, data_I=data_I, **kwargs)
        md_online.meta_task_l = preprocess(
            md_online.meta_task_l,
            factor_num=self.factor_num,
            is_mlp=self.forecast_model == "MLP",
            alpha=self.alpha,
            step=self.step,
            H=self.horizon if self.data_dir == "us_data" else (1 + self.horizon),
            not_sequence=self.not_sequence,
            to_tensor=self.preprocess_tensor
        )
        return md_offline, md_online

    def offline_training(self, seed=43):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # with R.start(experiment_name=self.meta_exp_name):
        if self.naive:
            batch_size = 5000
            if self.market == "csi100":
                batch_size = 2000
            elif self.market == "csi500":
                batch_size = 8000
            bm = Benchmark(
                data_dir=self.data_dir,
                market=self.market,
                model_type=self.forecast_model,
                alpha=self.alpha,
                rank_label=self.rank_label,
                h_path=self.h_path,
                task_ext_conf={'model': {'kwargs': {'batch_size': batch_size}}},
                init_data=False,
            )
            R.set_uri("../../benchmarks/mlruns/")
            model = bm.get_fitted_model(f"_{seed}")
            R.set_uri("./mlruns/")
        else:
            model = None

        if self.naive:
            mm = MetaModelInc(self.basic_task, is_rnn=self.is_rnn, x_dim=self.x_dim, lr_model=self.lr_model,
                              first_order=self.first_order, alpha=self.alpha, pretrained_model=model,
                              begin_valid_epoch=self.begin_valid_epoch)
        else:
            mm = DoubleAdaptManager(self.basic_task, is_rnn=self.is_rnn, x_dim=self.x_dim, lr_model=self.lr_model,
                                    first_order=self.first_order, alpha=self.alpha, pretrained_model=model,
                                    begin_valid_epoch=self.begin_valid_epoch, factor_num=self.factor_num,
                                    lr_da=self.lr, lr_ma=self.lr_model,
                                    adapt_x=self.adapt_x, adapt_y=self.adapt_y, reg=self.reg,
                                    num_head=self.num_head, temperature=self.temperature)
        if not self.naive:
            mm.fit(self.meta_dataset_offline)
            if self.save:
                print(f'Save checkpoint in Exp: {self.meta_exp_name + "_checkpoint"}')
                with R.start(experiment_name=self.meta_exp_name + "_checkpoint"):
                    R.save_objects(**{"framework": mm})

        # if self.naive and model is None:
        #     bm = Benchmark(
        #         data_dir=self.data_dir,
        #         market=self.market,
        #         model_type=self.forecast_model,
        #         alpha=self.alpha,
        #         rank_label=self.rank_label,
        #         h_path=self.h_path,
        #         task_ext_conf={'model': {'kwargs': {'batch_size': batch_size}}},
        #     )
        #     R.set_uri("../../benchmarks/mlruns/")
        #     with R.start(experiment_name=bm.exp_name + f"_{seed}"):
        #         model = init_instance_by_config(bm.basic_task()["model"])
        #         model.model = mm.framework.model
        #         model.fitted = True
        #         R.save_objects(**{"params.pkl": model})
        #     R.set_uri("./mlruns/")
        return mm

    def online_training(self, meta_tasks_test, meta_model, tag=""):
        ta = TimeAdjuster(future=True)
        segments = self.basic_task["dataset"]["kwargs"]["segments"]
        test_begin, test_end = ta.align_seg(segments["test"])
        print('Test segment:', test_begin, test_end)

        self.infer_exp_name = self.meta_exp_name + "_online" + tag
        with R.start(experiment_name=self.infer_exp_name):
            ds = init_instance_by_config(self.basic_task["dataset"], accept_types=Dataset)
            label_all = ds.prepare(segments="test", col_set="label", data_key=DataHandlerLP.DK_R)
            if isinstance(label_all, TSDataSampler):
                label_all = pd.DataFrame({"label": label_all.data_arr[:-1][:, 0]}, index=label_all.data_index)
                label_all = label_all.loc[test_begin:test_end]
            label_all = label_all.dropna(axis=0)
            mlp158 = self.forecast_model == "MLP" and self.alpha == 158
            pred_y_all, losses = meta_model.inference(meta_tasks_test)
            # tasks = []
            # for loss, task in zip(losses, meta_tasks_test.meta_task_l):
            #     segments = task.task["dataset"]["kwargs"]["segments"]
            #     tasks.append({'loss': loss, 'train': segments['train'], 'test': segments['test']})
            # R.save_objects(**{'task_list': tasks})
            if mlp158:
                pred_y_all = pred_y_all.loc[test_begin:test_end]
                label_all = label_all.loc[pred_y_all.index]
            else:
                pred_y_all = pred_y_all.loc[label_all.index]
            R.save_objects(**{"pred.pkl": pred_y_all[["pred"]], "label.pkl": label_all})
            # pred_y_all['label'] = label_all
            # K = 50
            # precision = pred_y_all.groupby(level='datetime').apply(
            #     lambda x: x['pred'].nlargest(K).index.isin(x['label'].nlargest(K).index).sum() / K).mean()
            # print('Precision@{}: {}'.format(K, precision))
            # R.log_metrics(**{'Precision': precision})
        rec = self.backtest(pred_y_all)
        return rec

    def backtest(self, pred_y_all):
        backtest_config = {
            "strategy": {
                "class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy",
                "kwargs": {"signal": "<PRED>", "topk": 50, "n_drop": 5},
            },
            "backtest": {
                "start_time": None,
                "end_time": None,
                "account": 100000000,
                "benchmark": self.benchmark,
                "exchange_kwargs": {
                    "limit_threshold": None if self.data_dir == "us_data" else 0.095,
                    "deal_price": "close",
                    "open_cost": 0.0005,
                    "close_cost": 0.0015,
                    "min_cost": 5,
                },
            },
        }
        rec = R.get_exp(experiment_name=self.infer_exp_name).list_recorders(rtype=Experiment.RT_L)[0]
        # mse = ((pred_y_all['pred'].to_numpy() - pred_y_all['label'].to_numpy()) ** 2).mean()
        # mae = np.abs(pred_y_all['pred'].to_numpy() - pred_y_all['label'].to_numpy()).mean()
        # print('mse:', mse, 'mae', mae)
        # rec.log_metrics(mse=mse, mae=mae)
        SigAnaRecord(recorder=rec, skip_existing=True).generate()
        PortAnaRecord(recorder=rec, config=backtest_config, skip_existing=True).generate()
        print(f"Your evaluation results can be found in the experiment named `{self.infer_exp_name}`.")
        return rec

    def run_all(self):
        self.meta_dataset_offline, self.meta_dataset_online = self.dump_data()
        all_metrics = {
            k: []
            for k in [
                # 'mse', 'mae',
                "IC",
                "ICIR",
                "Rank IC",
                "Rank ICIR",
                "1day.excess_return_with_cost.annualized_return",
                "1day.excess_return_with_cost.information_ratio",
                # "1day.excess_return_with_cost.max_drawdown",
            ]
        }
        # if self.rank_label:
        #     all_metrics.pop('IC')
        #     all_metrics.pop('ICIR')
        train_time = []
        test_time = []
        if not self.tag:
            self.tag = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
        for i in range(0, 10):
            start_time = time.time()
            np.random.seed(i)
            try:
                assert self.reload_tag is not None
                self.tag = self.reload_tag
                rec = R.get_exp(experiment_name=self.meta_exp_name + '_checkpoint').list_recorders(rtype=Experiment.RT_L)[i]
                mm: MetaModelInc = rec.load_object("framework")
                mm.framework.lr_model = 0.001
                mm.framework.opt.param_groups[0]['lr'] = self.lr_model
                mm.opt.param_groups[0]['lr'] = self.lr
                print('Reload experiment', self.meta_exp_name + '_checkpoint')
            except Exception as e:
                traceback.print_exc()
                print('No valid experiment to reload. Restart offline training...')
                mm = self.offline_training(seed=43 + i)
            train_time.append(time.time() - start_time)
            start_time = time.time()
            rec = self.online_training(self.meta_dataset_online, mm)
            test_time.append(time.time() - start_time)
            # exp = R.get_exp(experiment_name=self.infer_exp_name)
            # rec = exp.list_recorders(rtype=exp.RT_L)[0]
            metrics = rec.list_metrics()
            for k in all_metrics.keys():
                all_metrics[k].append(metrics[k])
            pprint(all_metrics)

        with R.start(experiment_name=self.meta_exp_name + "_final"):
            R.save_objects(all_metrics=all_metrics)
            train_time, test_time = np.array(train_time), np.array(test_time)
            R.log_metrics(train_time=train_time, test_time=train_time)
            print(f"Time cost: {train_time.mean()}\t{test_time.mean()}")
            res = {}
            for k in all_metrics.keys():
                v = np.array(all_metrics[k])
                res[k] = [v.mean(), v.std()]
                R.log_metrics(**{"final_" + k: res[k][0]})
                R.log_metrics(**{"final_" + k + "_std": res[k][1]})
            pprint(res)


if __name__ == "__main__":
    print(sys.argv)
    fire.Fire(Incremental)
