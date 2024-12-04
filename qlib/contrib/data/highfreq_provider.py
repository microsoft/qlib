import os
import time
import datetime
from typing import Optional

import qlib
from qlib import get_module_logger
from qlib.data import D
from qlib.config import REG_CN
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.data import Cal
from qlib.contrib.ops.high_freq import get_calendar_day, DayLast, FFillNan, BFillNan, Date, Select, IsNull, IsInf, Cut
import pickle as pkl
from joblib import Parallel, delayed


class HighFreqProvider:
    def __init__(
        self,
        start_time: str,
        end_time: str,
        train_end_time: str,
        valid_start_time: str,
        valid_end_time: str,
        test_start_time: str,
        qlib_conf: dict,
        feature_conf: dict,
        label_conf: Optional[dict] = None,
        backtest_conf: dict = None,
        freq: str = "1min",
        **kwargs,
    ) -> None:
        self.start_time = start_time
        self.end_time = end_time
        self.test_start_time = test_start_time
        self.train_end_time = train_end_time
        self.valid_start_time = valid_start_time
        self.valid_end_time = valid_end_time
        self._init_qlib(qlib_conf)
        self.feature_conf = feature_conf
        self.label_conf = label_conf
        self.backtest_conf = backtest_conf
        self.qlib_conf = qlib_conf
        self.logger = get_module_logger("HighFreqProvider")
        self.freq = freq

    def get_pre_datasets(self):
        """Generate the training, validation and test datasets for prediction

        Returns:
            Tuple[BaseDataset, BaseDataset, BaseDataset]: The training and test datasets
        """

        dict_feature_path = self.feature_conf["path"]
        train_feature_path = dict_feature_path[:-4] + "_train.pkl"
        valid_feature_path = dict_feature_path[:-4] + "_valid.pkl"
        test_feature_path = dict_feature_path[:-4] + "_test.pkl"

        dict_label_path = self.label_conf["path"]
        train_label_path = dict_label_path[:-4] + "_train.pkl"
        valid_label_path = dict_label_path[:-4] + "_valid.pkl"
        test_label_path = dict_label_path[:-4] + "_test.pkl"

        if (
            not os.path.isfile(train_feature_path)
            or not os.path.isfile(valid_feature_path)
            or not os.path.isfile(test_feature_path)
        ):
            xtrain, xvalid, xtest = self._gen_data(self.feature_conf)
            xtrain.to_pickle(train_feature_path)
            xvalid.to_pickle(valid_feature_path)
            xtest.to_pickle(test_feature_path)
            del xtrain, xvalid, xtest

        if (
            not os.path.isfile(train_label_path)
            or not os.path.isfile(valid_label_path)
            or not os.path.isfile(test_label_path)
        ):
            ytrain, yvalid, ytest = self._gen_data(self.label_conf)
            ytrain.to_pickle(train_label_path)
            yvalid.to_pickle(valid_label_path)
            ytest.to_pickle(test_label_path)
            del ytrain, yvalid, ytest

        feature = {
            "train": train_feature_path,
            "valid": valid_feature_path,
            "test": test_feature_path,
        }

        label = {
            "train": train_label_path,
            "valid": valid_label_path,
            "test": test_label_path,
        }

        return feature, label

    def get_backtest(self, **kwargs) -> None:
        self._gen_data(self.backtest_conf)

    def _init_qlib(self, qlib_conf):
        """initialize qlib"""

        qlib.init(
            region=REG_CN,
            auto_mount=False,
            custom_ops=[DayLast, FFillNan, BFillNan, Date, Select, IsNull, IsInf, Cut],
            expression_cache=None,
            **qlib_conf,
        )

    def _prepare_calender_cache(self):
        """preload the calendar for cache"""

        # This code used the copy-on-write feature of Linux
        # to avoid calculating the calendar multiple times in the subprocess.
        # This code may accelerate, but may be not useful on Windows and Mac Os
        Cal.calendar(freq=self.freq)
        get_calendar_day(freq=self.freq)

    def _gen_dataframe(self, config, datasets=["train", "valid", "test"]):
        try:
            path = config.pop("path")
        except KeyError as e:
            raise ValueError("Must specify the path to save the dataset.") from e
        if os.path.isfile(path):
            start = time.time()
            self.logger.info(f"[{__name__}]Dataset exists, load from disk.")

            # res = dataset.prepare(['train', 'valid', 'test'])
            with open(path, "rb") as f:
                data = pkl.load(f)
            if isinstance(data, dict):
                res = [data[i] for i in datasets]
            else:
                res = data.prepare(datasets)
            self.logger.info(f"[{__name__}]Data loaded, time cost: {time.time() - start:.2f}")
        else:
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            self.logger.info(f"[{__name__}]Generating dataset")
            start_time = time.time()
            self._prepare_calender_cache()
            dataset = init_instance_by_config(config)
            trainset, validset, testset = dataset.prepare(["train", "valid", "test"])
            data = {
                "train": trainset,
                "valid": validset,
                "test": testset,
            }
            with open(path, "wb") as f:
                pkl.dump(data, f)
            with open(path[:-4] + "train.pkl", "wb") as f:
                pkl.dump(trainset, f)
            with open(path[:-4] + "valid.pkl", "wb") as f:
                pkl.dump(validset, f)
            with open(path[:-4] + "test.pkl", "wb") as f:
                pkl.dump(testset, f)
            res = [data[i] for i in datasets]
            self.logger.info(f"[{__name__}]Data generated, time cost: {(time.time() - start_time):.2f}")
        return res

    def _gen_data(self, config, datasets=["train", "valid", "test"]):
        try:
            path = config.pop("path")
        except KeyError as e:
            raise ValueError("Must specify the path to save the dataset.") from e
        if os.path.isfile(path):
            start = time.time()
            self.logger.info(f"[{__name__}]Dataset exists, load from disk.")

            # res = dataset.prepare(['train', 'valid', 'test'])
            with open(path, "rb") as f:
                data = pkl.load(f)
            if isinstance(data, dict):
                res = [data[i] for i in datasets]
            else:
                res = data.prepare(datasets)
            self.logger.info(f"[{__name__}]Data loaded, time cost: {time.time() - start:.2f}")
        else:
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            self.logger.info(f"[{__name__}]Generating dataset")
            start_time = time.time()
            self._prepare_calender_cache()
            dataset = init_instance_by_config(config)
            dataset.config(dump_all=True, recursive=True)
            dataset.to_pickle(path)
            res = dataset.prepare(datasets)
            self.logger.info(f"[{__name__}]Data generated, time cost: {(time.time() - start_time):.2f}")
        return res

    def _gen_dataset(self, config):
        try:
            path = config.pop("path")
        except KeyError as e:
            raise ValueError("Must specify the path to save the dataset.") from e
        if os.path.isfile(path):
            start = time.time()
            self.logger.info(f"[{__name__}]Dataset exists, load from disk.")

            with open(path, "rb") as f:
                dataset = pkl.load(f)
            self.logger.info(f"[{__name__}]Data loaded, time cost: {time.time() - start:.2f}")
        else:
            start = time.time()
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            self.logger.info(f"[{__name__}]Generating dataset")
            self._prepare_calender_cache()
            dataset = init_instance_by_config(config)
            self.logger.info(f"[{__name__}]Dataset init, time cost: {time.time() - start:.2f}")
            dataset.prepare(["train", "valid", "test"])
            self.logger.info(f"[{__name__}]Dataset prepared, time cost: {time.time() - start:.2f}")
            dataset.config(dump_all=True, recursive=True)
            dataset.to_pickle(path)
        return dataset

    def _gen_day_dataset(self, config, conf_type):
        try:
            path = config.pop("path")
        except KeyError as e:
            raise ValueError("Must specify the path to save the dataset.") from e

        if os.path.isfile(path + "tmp_dataset.pkl"):
            start = time.time()
            self.logger.info(f"[{__name__}]Dataset exists, load from disk.")
        else:
            start = time.time()
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            self.logger.info(f"[{__name__}]Generating dataset")
            self._prepare_calender_cache()
            dataset = init_instance_by_config(config)
            self.logger.info(f"[{__name__}]Dataset init, time cost: {time.time() - start:.2f}")
            dataset.config(dump_all=False, recursive=True)
            dataset.to_pickle(path + "tmp_dataset.pkl")

        with open(path + "tmp_dataset.pkl", "rb") as f:
            new_dataset = pkl.load(f)

        time_list = D.calendar(start_time=self.start_time, end_time=self.end_time, freq=self.freq)[::240]

        def generate_dataset(times):
            if os.path.isfile(path + times.strftime("%Y-%m-%d") + ".pkl"):
                print("exist " + times.strftime("%Y-%m-%d"))
                return
            self._init_qlib(self.qlib_conf)
            end_times = times + datetime.timedelta(days=1)
            new_dataset.handler.config(**{"start_time": times, "end_time": end_times})
            if conf_type == "backtest":
                new_dataset.handler.setup_data()
            else:
                new_dataset.handler.setup_data(init_type=DataHandlerLP.IT_LS)
            new_dataset.config(dump_all=True, recursive=True)
            new_dataset.to_pickle(path + times.strftime("%Y-%m-%d") + ".pkl")

        Parallel(n_jobs=8)(delayed(generate_dataset)(times) for times in time_list)

    def _gen_stock_dataset(self, config, conf_type):
        try:
            path = config.pop("path")
        except KeyError as e:
            raise ValueError("Must specify the path to save the dataset.") from e

        if os.path.isfile(path + "tmp_dataset.pkl"):
            start = time.time()
            self.logger.info(f"[{__name__}]Dataset exists, load from disk.")
        else:
            start = time.time()
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            self.logger.info(f"[{__name__}]Generating dataset")
            self._prepare_calender_cache()
            dataset = init_instance_by_config(config)
            self.logger.info(f"[{__name__}]Dataset init, time cost: {time.time() - start:.2f}")
            dataset.config(dump_all=False, recursive=True)
            dataset.to_pickle(path + "tmp_dataset.pkl")

        with open(path + "tmp_dataset.pkl", "rb") as f:
            new_dataset = pkl.load(f)

        instruments = D.instruments(market="all")
        stock_list = D.list_instruments(
            instruments=instruments, start_time=self.start_time, end_time=self.end_time, freq=self.freq, as_list=True
        )

        def generate_dataset(stock):
            if os.path.isfile(path + stock + ".pkl"):
                print("exist " + stock)
                return
            self._init_qlib(self.qlib_conf)
            new_dataset.handler.config(**{"instruments": [stock]})
            if conf_type == "backtest":
                new_dataset.handler.setup_data()
            else:
                new_dataset.handler.setup_data(init_type=DataHandlerLP.IT_LS)
            new_dataset.config(dump_all=True, recursive=True)
            new_dataset.to_pickle(path + stock + ".pkl")

        Parallel(n_jobs=32)(delayed(generate_dataset)(stock) for stock in stock_list)
