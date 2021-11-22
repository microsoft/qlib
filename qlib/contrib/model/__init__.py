# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
try:
    from .catboost_model import CatBoostModel
except ModuleNotFoundError:
    CatBoostModel = None
    print("Please install necessary libs for CatBoostModel.")
try:
    from .double_ensemble import DEnsembleModel
    from .gbdt import LGBModel
except ModuleNotFoundError:
    DEnsembleModel, LGBModel = None, None
    print("Please install necessary libs for DEnsembleModel and LGBModel, such as lightgbm.")
try:
    from .xgboost import XGBModel
except ModuleNotFoundError:
    XGBModel = None
    print("Please install necessary libs for XGBModel, such as xgboost.")
try:
    from .linear import LinearModel
except ModuleNotFoundError:
    LinearModel = None
    print("Please install necessary libs for LinearModel, such as scipy and sklearn.")
# import pytorch models
try:
    from .pytorch_alstm import ALSTM
    from .pytorch_gats import GATs
    from .pytorch_gru import GRU
    from .pytorch_lstm import LSTM
    from .pytorch_nn import DNNModelPytorch
    from .pytorch_tabnet import TabnetModel
    from .pytorch_sfm import SFM_Model
    from .pytorch_tcn import TCN
    from .pytorch_add import ADD

    pytorch_classes = (ALSTM, GATs, GRU, LSTM, DNNModelPytorch, TabnetModel, SFM_Model, TCN, ADD)
except ModuleNotFoundError:
    pytorch_classes = ()
    print("Please install necessary libs for PyTorch models.")

all_model_classes = (CatBoostModel, DEnsembleModel, LGBModel, XGBModel, LinearModel) + pytorch_classes
