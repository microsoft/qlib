# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Collector module can collect objects from everywhere and process them such as merging, grouping, averaging and so on.
"""

from collections import defaultdict
from qlib.log import TimeInspector
from typing import Callable, Dict, Iterable, List
from qlib.log import get_module_logger
from qlib.utils.serial import Serializable
from qlib.utils.exceptions import LoadObjectError
from qlib.workflow import R
from qlib.workflow.exp import Experiment
from qlib.workflow.recorder import Recorder


class Collector(Serializable):
    """The collector to collect different results"""

    pickle_backend = "dill"  # use dill to dump user method

    def __init__(self, process_list=[]):
        """
        Init Collector.

        Args:
            process_list (list or Callable):  the list of processors or the instance of a processor to process dict.
        """
        if not isinstance(process_list, list):
            process_list = [process_list]
        self.process_list = process_list

    def collect(self) -> dict:
        """
        Collect the results and return a dict like {key: things}

        Returns:
            dict: the dict after collecting.

            For example:

            {"prediction": pd.Series}

            {"IC": {"Xgboost": pd.Series, "LSTM": pd.Series}}

            ...
        """
        raise NotImplementedError(f"Please implement the `collect` method.")

    @staticmethod
    def process_collect(collected_dict, process_list=[], *args, **kwargs) -> dict:
        """
        Do a series of processing to the dict returned by collect and return a dict like {key: things}
        For example, you can group and ensemble.

        Args:
            collected_dict (dict): the dict return by `collect`
            process_list (list or Callable): the list of processors or the instance of a processor to process dict.
                The processor order is the same as the list order.
                For example: [Group1(..., Ensemble1()), Group2(..., Ensemble2())]

        Returns:
            dict: the dict after processing.
        """
        if not isinstance(process_list, list):
            process_list = [process_list]
        result = {}
        for artifact in collected_dict:
            value = collected_dict[artifact]
            for process in process_list:
                if not callable(process):
                    raise NotImplementedError(f"{type(process)} is not supported in `process_collect`.")
                value = process(value, *args, **kwargs)
            result[artifact] = value
        return result

    def __call__(self, *args, **kwargs) -> dict:
        """
        Do the workflow including ``collect`` and ``process_collect``

        Returns:
            dict: the dict after collecting and processing.
        """
        collected = self.collect()
        return self.process_collect(collected, self.process_list, *args, **kwargs)


class MergeCollector(Collector):
    """
    A collector to collect the results of other Collectors

    For example:

        We have 2 collector, which named A and B.
        A can collect {"prediction": pd.Series} and B can collect {"IC": {"Xgboost": pd.Series, "LSTM": pd.Series}}.
        Then after this class's collect, we can collect {"A_prediction": pd.Series, "B_IC": {"Xgboost": pd.Series, "LSTM": pd.Series}}

        ...

    """

    def __init__(self, collector_dict: Dict[str, Collector], process_list: List[Callable] = [], merge_func=None):
        """
        Init MergeCollector.

        Args:
            collector_dict (Dict[str,Collector]): the dict like {collector_key, Collector}
            process_list (List[Callable]): the list of processors or the instance of processor to process dict.
            merge_func (Callable): a method to generate outermost key. The given params are ``collector_key`` from collector_dict and ``key`` from every collector after collecting.
                None for using tuple to connect them, such as "ABC"+("a","b") -> ("ABC", ("a","b")).
        """
        super().__init__(process_list=process_list)
        self.collector_dict = collector_dict
        self.merge_func = merge_func

    def collect(self) -> dict:
        """
        Collect all results of collector_dict and change the outermost key to a recombination key.

        Returns:
            dict: the dict after collecting.
        """
        collect_dict = {}
        for collector_key, collector in self.collector_dict.items():
            tmp_dict = collector()
            for key, value in tmp_dict.items():
                if self.merge_func is not None:
                    collect_dict[self.merge_func(collector_key, key)] = value
                else:
                    collect_dict[(collector_key, key)] = value
        return collect_dict


class RecorderCollector(Collector):
    ART_KEY_RAW = "__raw"

    def __init__(
        self,
        experiment,
        process_list=[],
        rec_key_func=None,
        rec_filter_func=None,
        artifacts_path={"pred": "pred.pkl"},
        artifacts_key=None,
        list_kwargs={},
        status: Iterable = {Recorder.STATUS_FI},
    ):
        """
        Init RecorderCollector.

        Args:
            experiment:
                (Experiment or str): an instance of an Experiment or the name of an Experiment
                (Callable): an callable function, which returns a list of experiments
            process_list (list or Callable): the list of processors or the instance of a processor to process dict.
            rec_key_func (Callable): a function to get the key of a recorder. If None, use recorder id.
            rec_filter_func (Callable, optional): filter the recorder by return True or False. Defaults to None.
            artifacts_path (dict, optional): The artifacts name and its path in Recorder. Defaults to {"pred": "pred.pkl", "IC": "sig_analysis/ic.pkl"}.
            artifacts_key (str or List, optional): the artifacts key you want to get. If None, get all artifacts.
            list_kwargs (str): arguments for list_recorders function.
            status (Iterable): only collect recorders with specific status. None indicating collecting all the recorders
        """
        super().__init__(process_list=process_list)
        if isinstance(experiment, str):
            experiment = R.get_exp(experiment_name=experiment)
        assert isinstance(experiment, (Experiment, Callable))
        self.experiment = experiment
        self.artifacts_path = artifacts_path
        if rec_key_func is None:

            def rec_key_func(rec):
                return rec.info["id"]

        if artifacts_key is None:
            artifacts_key = list(self.artifacts_path.keys())
        self.rec_key_func = rec_key_func
        self.artifacts_key = artifacts_key
        self.rec_filter_func = rec_filter_func
        self.list_kwargs = list_kwargs
        self.status = status

    def collect(self, artifacts_key=None, rec_filter_func=None, only_exist=True) -> dict:
        """
        Collect different artifacts based on recorder after filtering.

        Args:
            artifacts_key (str or List, optional): the artifacts key you want to get. If None, use the default.
            rec_filter_func (Callable, optional): filter the recorder by return True or False. If None, use the default.
            only_exist (bool, optional): if only collect the artifacts when a recorder really has.
                If True, the recorder with exception when loading will not be collected. But if False, it will raise the exception.

        Returns:
            dict: the dict after collected like {artifact: {rec_key: object}}
        """
        if artifacts_key is None:
            artifacts_key = self.artifacts_key
        if rec_filter_func is None:
            rec_filter_func = self.rec_filter_func

        if isinstance(artifacts_key, str):
            artifacts_key = [artifacts_key]

        collect_dict = {}
        # filter records

        if isinstance(self.experiment, Experiment):
            with TimeInspector.logt("Time to `list_recorders` in RecorderCollector"):
                recs = list(self.experiment.list_recorders(**self.list_kwargs).values())
        elif isinstance(self.experiment, Callable):
            recs = self.experiment()

        recs = [
            rec
            for rec in recs
            if (
                (self.status is None or rec.status in self.status) and (rec_filter_func is None or rec_filter_func(rec))
            )
        ]

        logger = get_module_logger("RecorderCollector")
        status_stat = defaultdict(int)
        for r in recs:
            status_stat[r.status] += 1
        logger.info(f"Nubmer of recorders after filter: {status_stat}")
        for rec in recs:
            rec_key = self.rec_key_func(rec)
            for key in artifacts_key:
                if self.ART_KEY_RAW == key:
                    artifact = rec
                else:
                    try:
                        artifact = rec.load_object(self.artifacts_path[key])
                    except LoadObjectError as e:
                        if only_exist:
                            # only collect existing artifact
                            logger.warning(f"Fail to load {self.artifacts_path[key]} and it is ignored.")
                            continue
                        raise e
                # give user some warning if the values are overridden
                cdd = collect_dict.setdefault(key, {})
                if rec_key in cdd:
                    logger.warning(
                        f"key '{rec_key}' is duplicated. Previous value will be overrides. Please check you `rec_key_func`"
                    )
                cdd[rec_key] = artifact

        return collect_dict

    def get_exp_name(self) -> str:
        """
        Get experiment name

        Returns:
            str: experiment name
        """
        return self.experiment.name
