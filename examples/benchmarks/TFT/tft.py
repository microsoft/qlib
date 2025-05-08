# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import data_formatters.base
import expt_settings.configs
import libs.hyperparam_opt
import libs.tft_model
import libs.utils as utils
import os
import datetime as dte


from qlib.model.base import ModelFT
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


# To register new datasets, please add them here.
ALLOW_DATASET = ["Alpha158", "Alpha360"]
# To register new datasets, please add their configurations here.
DATASET_SETTING = {
    "Alpha158": {
        "feature_col": [
            "RESI5",
            "WVMA5",
            "RSQR5",
            "KLEN",
            "RSQR10",
            "CORR5",
            "CORD5",
            "CORR10",
            "ROC60",
            "RESI10",
            "VSTD5",
            "RSQR60",
            "CORR60",
            "WVMA60",
            "STD5",
            "RSQR20",
            "CORD60",
            "CORD10",
            "CORR20",
            "KLOW",
        ],
        "label_col": "LABEL0",
    },
    "Alpha360": {
        "feature_col": [
            "HIGH0",
            "LOW0",
            "OPEN0",
            "CLOSE1",
            "HIGH1",
            "VOLUME1",
            "LOW1",
            "VOLUME3",
            "OPEN1",
            "VOLUME4",
            "CLOSE2",
            "CLOSE4",
            "VOLUME5",
            "LOW2",
            "CLOSE3",
            "VOLUME2",
            "HIGH2",
            "LOW4",
            "VOLUME8",
            "VOLUME11",
        ],
        "label_col": "LABEL0",
    },
}


def get_shifted_label(data_df, shifts=5, col_shift="LABEL0"):
    return data_df[[col_shift]].groupby("instrument", group_keys=False).apply(lambda df: df.shift(shifts))


def fill_test_na(test_df):
    test_df_res = test_df.copy()
    feature_cols = ~test_df_res.columns.str.contains("label", case=False)
    test_feature_fna = (
        test_df_res.loc[:, feature_cols].groupby("datetime", group_keys=False).apply(lambda df: df.fillna(df.mean()))
    )
    test_df_res.loc[:, feature_cols] = test_feature_fna
    return test_df_res


def process_qlib_data(df, dataset, fillna=False):
    """Prepare data to fit the TFT model.

    Args:
      df: Original DataFrame.
      fillna: Whether to fill the data with the mean values.

    Returns:
      Transformed DataFrame.

    """
    # Several features selected manually
    feature_col = DATASET_SETTING[dataset]["feature_col"]
    label_col = [DATASET_SETTING[dataset]["label_col"]]
    temp_df = df.loc[:, feature_col + label_col]
    if fillna:
        temp_df = fill_test_na(temp_df)
    temp_df = temp_df.swaplevel()
    temp_df = temp_df.sort_index()
    temp_df = temp_df.reset_index(level=0)
    dates = pd.to_datetime(temp_df.index)
    temp_df["date"] = dates
    temp_df["day_of_week"] = dates.dayofweek
    temp_df["month"] = dates.month
    temp_df["year"] = dates.year
    temp_df["const"] = 1.0
    return temp_df


def process_predicted(df, col_name):
    """Transform the TFT predicted data into Qlib format.

    Args:
      df: Original DataFrame.
      fillna: New column name.

    Returns:
      Transformed DataFrame.

    """
    df_res = df.copy()
    df_res = df_res.rename(columns={"forecast_time": "datetime", "identifier": "instrument", "t+4": col_name})
    df_res = df_res.set_index(["datetime", "instrument"]).sort_index()
    df_res = df_res[[col_name]]
    return df_res


def format_score(forecast_df, col_name="pred", label_shift=5):
    pred = process_predicted(forecast_df, col_name=col_name)
    pred = get_shifted_label(pred, shifts=-label_shift, col_shift=col_name)
    pred = pred.dropna()[col_name]
    return pred


def transform_df(df, col_name="LABEL0"):
    df_res = df["feature"]
    df_res[col_name] = df["label"]
    return df_res


class TFTModel(ModelFT):
    """TFT Model"""

    def __init__(self, **kwargs):
        self.model = None
        self.params = {"DATASET": "Alpha158", "label_shift": 5}
        self.params.update(kwargs)

    def _prepare_data(self, dataset: DatasetH):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        return transform_df(df_train), transform_df(df_valid)

    def fit(self, dataset: DatasetH, MODEL_FOLDER="qlib_tft_model", USE_GPU_ID=0, **kwargs):
        DATASET = self.params["DATASET"]
        LABEL_SHIFT = self.params["label_shift"]
        LABEL_COL = DATASET_SETTING[DATASET]["label_col"]

        if DATASET not in ALLOW_DATASET:
            raise AssertionError("The dataset is not supported, please make a new formatter to fit this dataset")

        dtrain, dvalid = self._prepare_data(dataset)
        dtrain.loc[:, LABEL_COL] = get_shifted_label(dtrain, shifts=LABEL_SHIFT, col_shift=LABEL_COL)
        dvalid.loc[:, LABEL_COL] = get_shifted_label(dvalid, shifts=LABEL_SHIFT, col_shift=LABEL_COL)

        train = process_qlib_data(dtrain, DATASET, fillna=True).dropna()
        valid = process_qlib_data(dvalid, DATASET, fillna=True).dropna()

        ExperimentConfig = expt_settings.configs.ExperimentConfig
        config = ExperimentConfig(DATASET)
        self.data_formatter = config.make_data_formatter()
        self.model_folder = MODEL_FOLDER
        self.gpu_id = USE_GPU_ID
        self.label_shift = LABEL_SHIFT
        self.expt_name = DATASET
        self.label_col = LABEL_COL

        use_gpu = (True, self.gpu_id)
        # ===========================Training Process===========================
        ModelClass = libs.tft_model.TemporalFusionTransformer
        if not isinstance(self.data_formatter, data_formatters.base.GenericDataFormatter):
            raise ValueError(
                "Data formatters should inherit from"
                + "AbstractDataFormatter! Type={}".format(type(self.data_formatter))
            )

        default_keras_session = tf.keras.backend.get_session()

        if use_gpu[0]:
            self.tf_config = utils.get_default_tensorflow_config(tf_device="gpu", gpu_id=use_gpu[1])
        else:
            self.tf_config = utils.get_default_tensorflow_config(tf_device="cpu")

        self.data_formatter.set_scalers(train)

        # Sets up default params
        fixed_params = self.data_formatter.get_experiment_params()
        params = self.data_formatter.get_default_model_params()

        params = {**params, **fixed_params}

        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
        params["model_folder"] = self.model_folder

        print("*** Begin training ***")
        best_loss = np.Inf

        tf.reset_default_graph()

        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.sess = tf.Session(config=self.tf_config)
            tf.keras.backend.set_session(self.sess)
            self.model = ModelClass(params, use_cudnn=use_gpu[0])
            self.sess.run(tf.global_variables_initializer())
            self.model.fit(train_df=train, valid_df=valid)
            print("*** Finished training ***")
            saved_model_dir = self.model_folder + "/" + "saved_model"
            if not os.path.exists(saved_model_dir):
                os.makedirs(saved_model_dir)
            self.model.save(saved_model_dir)

            def extract_numerical_data(data):
                """Strips out forecast time and identifier columns."""
                return data[[col for col in data.columns if col not in {"forecast_time", "identifier"}]]

            # p50_loss = utils.numpy_normalised_quantile_loss(
            #    extract_numerical_data(targets), extract_numerical_data(p50_forecast),
            #    0.5)
            # p90_loss = utils.numpy_normalised_quantile_loss(
            #    extract_numerical_data(targets), extract_numerical_data(p90_forecast),
            #    0.9)
            tf.keras.backend.set_session(default_keras_session)
        print("Training completed at {}.".format(dte.datetime.now()))
        # ===========================Training Process===========================

    def predict(self, dataset):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        d_test = dataset.prepare("test", col_set=["feature", "label"])
        d_test = transform_df(d_test)
        d_test.loc[:, self.label_col] = get_shifted_label(d_test, shifts=self.label_shift, col_shift=self.label_col)
        test = process_qlib_data(d_test, self.expt_name, fillna=True).dropna()

        use_gpu = (True, self.gpu_id)
        # ===========================Predicting Process===========================
        default_keras_session = tf.keras.backend.get_session()

        # Sets up default params
        fixed_params = self.data_formatter.get_experiment_params()
        params = self.data_formatter.get_default_model_params()
        params = {**params, **fixed_params}

        print("*** Begin predicting ***")
        tf.reset_default_graph()

        with self.tf_graph.as_default():
            tf.keras.backend.set_session(self.sess)
            output_map = self.model.predict(test, return_targets=True)
            targets = self.data_formatter.format_predictions(output_map["targets"])
            p50_forecast = self.data_formatter.format_predictions(output_map["p50"])
            p90_forecast = self.data_formatter.format_predictions(output_map["p90"])
            tf.keras.backend.set_session(default_keras_session)

        predict50 = format_score(p50_forecast, "pred", 1)
        predict90 = format_score(p90_forecast, "pred", 1)
        predict = (predict50 + predict90) / 2  # self.label_shift
        # ===========================Predicting Process===========================
        return predict

    def finetune(self, dataset: DatasetH):
        """
        finetune model
        Parameters
        ----------
        dataset : DatasetH
            dataset for finetuning
        """
        pass

    def to_pickle(self, path: Union[Path, str]):
        """
        Tensorflow model can't be dumped directly.
        So the data should be save separately

        **TODO**: Please implement the function to load the files

        Parameters
        ----------
        path : Union[Path, str]
            the target path to be dumped
        """
        # FIXME: implementing saving tensorflow models
        # save tensorflow model
        # path = Path(path)
        # path.mkdir(parents=True)
        # self.model.save(path)

        # save qlib model wrapper
        drop_attrs = ["model", "tf_graph", "sess", "data_formatter"]
        orig_attr = {}
        for attr in drop_attrs:
            orig_attr[attr] = getattr(self, attr)
            setattr(self, attr, None)
        super(TFTModel, self).to_pickle(path)
        for attr in drop_attrs:
            setattr(self, attr, orig_attr[attr])
