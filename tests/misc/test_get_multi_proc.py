#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import shutil
import unittest
from pathlib import Path

import qlib
from qlib.data import D
from qlib.tests.data import GetData
from qlib.tests import TestAutoData
from multiprocessing import Pool


class TestGetData(TestAutoData):
    FIELDS = "$open,$close,$high,$low,$volume,$factor,$change".split(",")

    def test_multi_proc(self):
        """
        For testing if it will raise error
        """
        qlib.init(provider_uri=TestAutoData.provider_uri, expression_cache=None, dataset_cache=None)
        args = D.instruments("csi300"), self.FIELDS

        iter_n = 2
        pool = Pool(iter_n)

        res = []
        for _ in range(iter_n):
            res.append(pool.apply_async(D.features, args, {}))

        for r in res:
            print(r.get())

        pool.close()
        pool.join()


if __name__ == "__main__":
    unittest.main()
