from qlib.model.base import Model
from typing import Text, Union
from qlib.data.dataset import DatasetH, Dataset
from tensorflow.keras import layers, models, optimizers, regularizers, losses, metrics
from qlib.data.dataset.handler import DataHandlerLP
import pandas as pd


class CNN(Model):
    def __init__(self,
                 epochs=30,
                 kernel_size=2,
                 conv_strides=1,
                 pool_size=3,
                 pool_strides=2,
                 l2=0.0003,
                 dropout=0.5,
                 lr=0.0001,
                 batch_size=128,
                 model=None,
                 dtrain=None,
                 dvalid=None,
                 dtest=None,
                 loss=losses.mean_squared_error):
        super(CNN, self).__init__()
        self.cnn_model = model
        self.kernel_size = kernel_size
        self.conv_strides = conv_strides
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.l2 = l2
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.dtrain = dtrain
        self.dvalid = dvalid
        self.dtest = dtest
        self.epochs = epochs
        self.loss = loss

    def fit(self, dataset: Dataset):
        if self.dtrain is None:
            dtrain, dvalid = dataset.prepare(
                ["train", "valid"],
                col_set=["feature", "label"],
                data_key=DataHandlerLP.DK_L,
            )
            self.dtrain = dtrain.fillna(method='bfill', axis=0).fillna(0)
            self.dvalid = dvalid.fillna(method='bfill', axis=0).fillna(0)

        x_train, y_train = self.dtrain["feature"], self.dtrain["label"]
        x_valid, y_valid = self.dvalid["feature"], self.dvalid["label"]

        x_train = x_train.values.reshape(-1, x_train.shape[1], 1)
        x_valid = x_valid.values.reshape(-1, x_valid.shape[1], 1)

        if self.cnn_model is None:
            rmse = metrics.RootMeanSquaredError(name='rmse')
            self.cnn_model = models.Sequential([
                layers.Conv1D(32, kernel_size=self.kernel_size, strides=self.conv_strides, activation='leaky_relu',
                              kernel_regularizer=regularizers.l2(self.l2)),
                layers.MaxPooling1D(pool_size=self.pool_size, strides=self.pool_strides),
                layers.Conv1D(64, kernel_size=self.kernel_size, strides=self.conv_strides,
                              kernel_regularizer=regularizers.l2(self.l2)),
                layers.LayerNormalization(),
                layers.LeakyReLU(0.2),
                layers.MaxPooling1D(pool_size=self.pool_size, strides=self.pool_strides),
                layers.Conv1D(64, kernel_size=self.kernel_size, strides=self.conv_strides, activation='leaky_relu',
                              kernel_regularizer=regularizers.l2(self.l2)),
                layers.Conv1D(128, kernel_size=self.kernel_size, strides=self.conv_strides,
                              kernel_regularizer=regularizers.l2(self.l2)),
                layers.LayerNormalization(),
                layers.LeakyReLU(0.2),
                layers.Conv1D(128, kernel_size=self.kernel_size, strides=self.conv_strides, activation='leaky_relu',
                              kernel_regularizer=regularizers.l2(self.l2)),
                layers.MaxPooling1D(pool_size=self.pool_size, strides=self.pool_strides),
                layers.Flatten(),
                layers.Dense(2048, activation='leaky_relu', kernel_regularizer=regularizers.l2(self.l2)),
                layers.Dropout(self.dropout),
                layers.Dense(1024, activation='leaky_relu', kernel_regularizer=regularizers.l2(self.l2)),
                layers.Dropout(self.dropout),
                layers.Dense(1024, activation='leaky_relu', kernel_regularizer=regularizers.l2(self.l2)),
                layers.Dropout(self.dropout),
                layers.Dense(1)
            ])
            self.cnn_model.compile(optimizer=optimizers.Adam(self.lr), loss=self.loss, metrics=[rmse, 'mae', 'mape'])
            self.cnn_model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                               validation_data=(x_valid, y_valid), verbose=1)
        self.cnn_model.summary()

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test") -> object:
        if self.dtest is not None:
            x_test, y_test = self.dtest['feature'], self.dtest['label']
        else:
            x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
            x_test = x_test.fillna(method='bfill', axis=0).fillna(0)

        index = x_test.index
        values = x_test.values
        values = values.reshape(-1, x_test.shape[1], 1)
        pred = list(self.cnn_model.predict(values))
        return pd.Series(pred, index=index)
