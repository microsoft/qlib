# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Collector can collect object from everywhere and process them such as merging, grouping, averaging and so on.
"""

from qlib.model.ens.ensemble import SingleKeyEnsemble
from qlib.workflow import R
import dill as pickle


class Collector:
    """The collector to collect different results"""

    def __init__(self, process_list=[]):
        """
        Args:
            process_list (list, optional): process_list (list or Callable): the list of processors or the instance of processor to process dict.
        """
        if not isinstance(process_list, list):
            process_list = [process_list]
        self.process_list = process_list

    def collect(self) -> dict:
        """Collect the results and return a dict like {key: things}

        Returns:
            dict: the dict after collecting.

            For example:

            {"prediction": pd.Series}

            {"IC": {"Xgboost": pd.Series, "LSTM": pd.Series}}

            ......
        """
        raise NotImplementedError(f"Please implement the `collect` method.")

    @staticmethod
    def process_collect(collected_dict, process_list=[], *args, **kwargs) -> dict:
        """do a series of processing to the dict returned by collect and return a dict like {key: things}
        For example: you can group and ensemble.

        Args:
            collected_dict (dict): the dict return by `collect`
            process_list (list or Callable): the list of processors or the instance of processor to process dict.
            The processor order is same as the list order.
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
        do the workflow including collect and process_collect

        Returns:
            dict: the dict after collecting and processing.
        """
        collected = self.collect()
        return self.process_collect(collected, self.process_list, *args, **kwargs)

    def save(self, filepath):
        """
        save the collector into a file

        Args:
            filepath (str): the path of file

        Returns:
            bool: if succeeded
        """
        try:
            with open(filepath, "wb") as f:
                pickle.dump(self, f)
        except Exception:
            return False
        return True

    @staticmethod
    def load(filepath):
        """
        load the collector from a file

        Args:
            filepath (str): the path of file

        Raises:
            TypeError: the pickled file must be `Collector`

        Returns:
            Collector: the instance of Collector
        """
        with open(filepath, "rb") as f:
            collector = pickle.load(f)
        if isinstance(collector, Collector):
            return collector
        else:
            raise TypeError(f"The instance of {type(collector)} is not a valid `Collector`!")


class HyperCollector(Collector):
    """
    A collector to collect the results of other Collectors
    """

    def __init__(self, collector_dict, process_list=[]):
        """
        Args:
            collector_dict (dict): the dict like {collector_key, Collector}
            process_list (list or Callable): the list of processors or the instance of processor to process dict.
                NOTE: process_list = [SingleKeyEnsemble()] can ignore key and use value directly if there is only one {k,v} in a dict.
                This can make result more readable. If you want to maintain as it should be, just give a empty process list.
        """
        super().__init__(process_list=process_list)
        self.collector_dict = collector_dict

    def collect(self) -> dict:
        collect_dict = {}
        for key, collector in self.collector_dict.items():
            collect_dict[key] = collector()
        return collect_dict


class RecorderCollector(Collector):
    ART_KEY_RAW = "__raw"

    def __init__(
        self,
        experiment,
        process_list=[],
        rec_key_func=None,
        rec_filter_func=None,
        artifacts_path={"pred": "pred.pkl", "IC": "sig_analysis/ic.pkl"},
        artifacts_key=None,
    ):
        """init RecorderCollector

        Args:
            experiment (Experiment or str): an instance of a Experiment or the name of a Experiment
            process_list (list or Callable): the list of processors or the instance of processor to process dict.
            rec_key_func (Callable): a function to get the key of a recorder. If None, use recorder id.
            rec_filter_func (Callable, optional): filter the recorder by return True or False. Defaults to None.
            artifacts_path (dict, optional): The artifacts name and its path in Recorder. Defaults to {"pred": "pred.pkl", "IC": "sig_analysis/ic.pkl"}.
            artifacts_key (str or List, optional): the artifacts key you want to get. If None, get all artifacts.
        """
        super().__init__(process_list=process_list)
        if isinstance(experiment, str):
            experiment = R.get_exp(experiment_name=experiment)
        self.experiment = experiment
        self.artifacts_path = artifacts_path
        if rec_key_func is None:
            rec_key_func = lambda rec: rec.info["id"]
        if artifacts_key is None:
            artifacts_key = list(self.artifacts_path.keys())
        self._rec_key_func = rec_key_func
        self.artifacts_key = artifacts_key
        self._rec_filter_func = rec_filter_func

    def collect(self, artifacts_key=None, rec_filter_func=None) -> dict:
        """Collect different artifacts based on recorder after filtering.

        Args:
            artifacts_key (str or List, optional): the artifacts key you want to get. If None, use default.
            rec_filter_func (Callable, optional): filter the recorder by return True or False. If None, use default.

        Returns:
            dict: the dict after collected like {artifact: {rec_key: object}}
        """
        if artifacts_key is None:
            artifacts_key = self.artifacts_key
        if rec_filter_func is None:
            rec_filter_func = self._rec_filter_func

        if isinstance(artifacts_key, str):
            artifacts_key = [artifacts_key]

        collect_dict = {}
        # filter records
        recs = self.experiment.list_recorders()
        recs_flt = {}
        for rid, rec in recs.items():
            if rec_filter_func is None or rec_filter_func(rec):
                recs_flt[rid] = rec

        for _, rec in recs_flt.items():
            rec_key = self._rec_key_func(rec)
            for key in artifacts_key:
                if self.ART_KEY_RAW == key:
                    artifact = rec
                else:
                    artifact = rec.load_object(self.artifacts_path[key])
                collect_dict.setdefault(key, {})[rec_key] = artifact

        return collect_dict

    def get_exp_name(self) -> str:
        """
        Get experiment name

        Returns:
            str: experiment name
        """
        return self.experiment.name
