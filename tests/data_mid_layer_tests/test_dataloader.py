
# TODO:
# dump alpha 360 to dataframe and merge it with Alpha158


import unittest

from qlib.data.dataset.loader import NestedDataLoader


class TestDataLoader(unittest.TestCase):


    def test_nested_data_loader(self):
        nd = NestedDataLoader(
            dataloader_l=[
                {
                    "class": "qlib.contrib.data.loader.Alpha158DL",
                }, {
                    "class": "qlib.contrib.data.loader.Alpha360DL",
                    "kwargs": {
                        "config": {
                            "label": ( ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"])
                        }
                    }
                }
            ]
        )
        # Of course you can use StaticDataLoader

        nd.load
        ...

        # Then you can use it wth DataHandler;


if __name__ == "__main__":
    unittest.main()
