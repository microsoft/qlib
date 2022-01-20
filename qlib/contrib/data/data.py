# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# We remove arctic from core framework of Qlib to contrib due to
# - Arctic has very strict limitation on pandas and numpy version
#    - https://github.com/man-group/arctic/pull/908
# - pip fail to computing the right version number!!!!
#    - Maybe we can solve this problem by poetry

# FIXME: So if you want to use arctic-based provider, please install arctic manually
# `pip install arctic` may not be enough.
from arctic import Arctic
import pandas as pd
import pymongo

from qlib.data.data import FeatureProvider


class ArcticFeatureProvider(FeatureProvider):
    def __init__(
        self, uri="127.0.0.1", retry_time=0, market_transaction_time_list=[("09:15", "11:30"), ("13:00", "15:00")]
    ):
        super().__init__()
        self.uri = uri
        # TODO:
        # retry connecting if error occurs
        # does it real matters?
        self.retry_time = retry_time
        # NOTE: this is especially important for TResample operator
        self.market_transaction_time_list = market_transaction_time_list

    def feature(self, instrument, field, start_index, end_index, freq):
        field = str(field)[1:]
        with pymongo.MongoClient(self.uri) as client:
            # TODO: this will result in frequently connecting the server and performance issue
            arctic = Arctic(client)

            if freq not in arctic.list_libraries():
                raise ValueError("lib {} not in arctic".format(freq))

            if instrument not in arctic[freq].list_symbols():
                # instruments does not exist
                return pd.Series()
            else:
                df = arctic[freq].read(instrument, columns=[field], chunk_range=(start_index, end_index))
                s = df[field]

                if not s.empty:
                    s = pd.concat(
                        [
                            s.between_time(time_tuple[0], time_tuple[1])
                            for time_tuple in self.market_transaction_time_list
                        ]
                    )
                return s
