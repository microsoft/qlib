# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pathlib import Path
from qlib.data.dataset.handler import DataHandlerLP

import pandas as pd
import fire
import sys
from tqdm.auto import tqdm
import yaml
from qlib import auto_init
from qlib.model.trainer import task_train
from qlib.utils import init_instance_by_config
from qlib.workflow.task.gen import RollingGen, task_generator

DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.append(str(DIRNAME.parent / "baseline"))
from rolling_benchmark import RollingBenchmark  # NOTE: sys.path is changed for import RollingBenchmark


class DDGDA:
    def __init__(self) -> None:
        self.step = 20

    def get_feature_importance(self):
        rb = RollingBenchmark()
        task = rb.basic_task()

        model = init_instance_by_config(task["model"])
        dataset = init_instance_by_config(task["dataset"])
        model.fit(dataset)

        fi = model.get_feature_importance()

        # Because the model use numpy instead of dataframe for training lightgbm
        # So the we must use following extra steps to get the right feature importance
        df = dataset.prepare(segments=slice(None), col_set="feature", data_key=DataHandlerLP.DK_R)
        cols = df.columns
        fi_named = {cols[int(k.split("_")[1])]: imp for k, imp in fi.to_dict().items()}

        return pd.Series(fi_named)

    def dump_data_for_proxy_model(self):
        """
            Dump data for training meta model.
            The meta model will be trained upon the proxy forecasting model.
        """
        topk = 30
        fi = self.get_feature_importance()
        col_selected = fi.nlargest(topk)

        rb = RollingBenchmark()
        task = rb.basic_task()
        dataset = init_instance_by_config(task["dataset"])
        prep_ds = dataset.prepare(slice(None), col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)

        feature_df = prep_ds["feature"]
        label_df = prep_ds["label"]

        feature_selected = feature_df.loc[:, col_selected.index]

        feature_selected = feature_selected.groupby("datetime").apply(lambda df: (df - df.mean()).div(df.std()))
        feature_selected = feature_selected.fillna(0.)

        df_all = {
            "label": label_df.reindex(feature_selected.index),
            "feature": feature_selected,
        }
        df_all = pd.concat(df_all, axis=1)
        df_all.to_pickle(DIRNAME / f"fea_label_df.pkl")

    def run_all(self):
        self.dump_data_for_proxy_model()


if __name__ == "__main__":
    auto_init()
    fire.Fire(DDGDA)
