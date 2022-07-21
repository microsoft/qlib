import unittest

from qlib.data import D
from qlib.data.dataset.loader import QlibDataLoader
from qlib.data.ops import ChangeInstrument, Cov, Feature, Ref, Var
from qlib.tests import TestOperatorData


class TestOperatorDataSetting(TestOperatorData):
    def test_setting(self):
        # All the query below passes
        df = D.features(["SH600519"], ["ChangeInstrument('SH000300', $close)"])

        # get market return for "SH600519"
        df = D.features(["SH600519"], ["ChangeInstrument('SH000300', Feature('close')/Ref(Feature('close'),1) -1)"])
        df = D.features(["SH600519"], ["ChangeInstrument('SH000300', $close/Ref($close,1) -1)"])
        # excess return
        df = D.features(
            ["SH600519"], ["($close/Ref($close,1) -1) - ChangeInstrument('SH000300', $close/Ref($close,1) -1)"]
        )
        print(df)

    def test_case2(self):
        def test_case(instruments, queries, note=None):
            if note:
                print(note)
            print(f"checking {instruments} with queries {queries}")
            df = D.features(instruments, queries)
            print(df)
            return df

        test_case(["SH600519"], ["ChangeInstrument('SH000300', $close)"], "get market index close")
        test_case(
            ["SH600519"],
            ["ChangeInstrument('SH000300', Feature('close')/Ref(Feature('close'),1) -1)"],
            "get market index return with Feature",
        )
        test_case(
            ["SH600519"],
            ["ChangeInstrument('SH000300', $close/Ref($close,1) -1)"],
            "get market index return with expression",
        )
        test_case(
            ["SH600519"],
            ["($close/Ref($close,1) -1) - ChangeInstrument('SH000300', $close/Ref($close,1) -1)"],
            "get excess return with expression with beta=1",
        )

        ret = "Feature('close') / Ref(Feature('close'), 1) - 1"
        benchmark = "SH000300"
        n_period = 252
        marketRet = f"ChangeInstrument('{benchmark}', Feature('close') / Ref(Feature('close'), 1) - 1)"
        marketVar = f"ChangeInstrument('{benchmark}', Var({marketRet}, {n_period}))"
        beta = f"Cov({ret}, {marketRet}, {n_period}) / {marketVar}"
        excess_return = f"{ret} - {beta}*({marketRet})"
        fields = [
            "Feature('close')",
            f"ChangeInstrument('{benchmark}', Feature('close'))",
            ret,
            marketRet,
            beta,
            excess_return,
        ]
        test_case(["SH600519"], fields[5:], "get market beta and excess_return with estimated beta")

        instrument = "sh600519"
        ret = Feature("close") / Ref(Feature("close"), 1) - 1
        benchmark = "sh000300"
        n_period = 252
        marketRet = ChangeInstrument(benchmark, Feature("close") / Ref(Feature("close"), 1) - 1)
        marketVar = ChangeInstrument(benchmark, Var(marketRet, n_period))
        beta = Cov(ret, marketRet, n_period) / marketVar
        fields = [
            Feature("close"),
            ChangeInstrument(benchmark, Feature("close")),
            ret,
            marketRet,
            beta,
            ret - beta * marketRet,
        ]
        names = ["close", "marketClose", "ret", "marketRet", f"beta_{n_period}", "excess_return"]
        data_loader_config = {"feature": (fields, names)}
        data_loader = QlibDataLoader(config=data_loader_config)
        df = data_loader.load(instruments=[instrument])  # , start_time=start_time)
        print(df)

        # test_case(["sh600519"],fields,
        # "get market beta and excess_return with estimated beta")


if __name__ == "__main__":
    unittest.main()
