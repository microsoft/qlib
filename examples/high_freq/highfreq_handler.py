from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from qlib.data.dataset.processor import Processor
from qlib.utils import get_cls_kwargs
from qlib.log import TimeInspector


class HighFreqHandler(DataHandlerLP):
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        freq="1min",
        infer_processors=[],
        learn_processors=[],
        fit_start_time=None,
        fit_end_time=None,
        drop_raw=True,
    ):
        def check_transform_proc(proc_l):
            new_l = []
            for p in proc_l:
                p["kwargs"].update(
                    {
                        "fit_start_time": fit_start_time,
                        "fit_end_time": fit_end_time,
                    }
                )
                new_l.append(p)
            return new_l

        infer_processors = check_transform_proc(infer_processors)
        learn_processors = check_transform_proc(learn_processors)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": self.get_feature_config(),
                "swap_level": False,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            freq=freq,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            drop_raw=drop_raw,
        )

    def get_feature_config(self):
        fields = []
        names = []

        template_if = "If(IsNull({1}), {0}, {1})"
        template_paused = "Select(Eq($paused, 0.0), {0})"
        # template_paused="{0}"
        template_fillnan = "FFillNan({0})"
        simpson_vwap = "($open + 2*$high + 2*$low + $close)/6"
        fields += [
            "{0}/Ref(DayLast({1}), 240)".format(
                template_if.format(
                    template_fillnan.format(template_paused.format("$close")),
                    template_paused.format("$open"),
                ),
                template_fillnan.format(template_paused.format("$close")),
            )
        ]
        fields += [
            "{0}/Ref(DayLast({1}), 240)".format(
                template_if.format(
                    template_fillnan.format(template_paused.format("$close")),
                    template_paused.format("$high"),
                ),
                template_fillnan.format(template_paused.format("$close")),
            )
        ]
        fields += [
            "{0}/Ref(DayLast({1}), 240)".format(
                template_if.format(
                    template_fillnan.format(template_paused.format("$close")),
                    template_paused.format("$low"),
                ),
                template_fillnan.format(template_paused.format("$close")),
            )
        ]
        fields += ["{0}/Ref(DayLast({0}), 240)".format(template_fillnan.format(template_paused.format("$close")))]
        fields += [
            "{0}/Ref(DayLast({1}), 240)".format(
                template_if.format(
                    template_fillnan.format(template_paused.format("$close")),
                    template_paused.format(simpson_vwap),
                ),
                template_fillnan.format(template_paused.format("$close")),
            )
        ]
        names += ["$open", "$high", "$low", "$close", "$vwap"]

        fields += [
            "Ref({0}, 240)/Ref(DayLast({1}), 240)".format(
                template_if.format(
                    template_fillnan.format(template_paused.format("$close")),
                    template_paused.format("$open"),
                ),
                template_fillnan.format(template_paused.format("$close")),
            )
        ]
        fields += [
            "Ref({0}, 240)/Ref(DayLast({1}), 240)".format(
                template_if.format(
                    template_fillnan.format(template_paused.format("$close")),
                    template_paused.format("$high"),
                ),
                template_fillnan.format(template_paused.format("$close")),
            )
        ]
        fields += [
            "Ref({0}, 240)/Ref(DayLast({1}), 240)".format(
                template_if.format(
                    template_fillnan.format(template_paused.format("$close")),
                    template_paused.format("$low"),
                ),
                template_fillnan.format(template_paused.format("$close")),
            )
        ]
        fields += [
            "Ref({0}, 240)/Ref(DayLast({0}), 240)".format(template_fillnan.format(template_paused.format("$close")))
        ]

        fields += [
            "Ref({0}, 240)/Ref(DayLast({1}), 240)".format(
                template_if.format(
                    template_fillnan.format(template_paused.format("$close")),
                    template_paused.format(simpson_vwap),
                ),
                template_fillnan.format(template_paused.format("$close")),
            )
        ]
        names += ["$open_1", "$high_1", "$low_1", "$close_1", "$vwap_1"]

        fields += [
            "{0}/Ref(DayLast(Mean({0}, 7200)), 240)".format(
                "If(IsNull({0}), 0, If(Or(Gt({1}, Mul(1.001, {3})), Lt({1}, Mul(0.999, {2}))), 0, {0}))".format(
                    template_paused.format("$volume"),
                    template_paused.format(simpson_vwap),
                    template_paused.format("$low"),
                    template_paused.format("$high"),
                )
            )
        ]
        names += ["$volume"]
        fields += [
            "Ref({0}, 240)/Ref(DayLast(Mean({0}, 7200)), 240)".format(
                "If(IsNull({0}), 0, If(Or(Gt({1}, Mul(1.001, {3})), Lt({1}, Mul(0.999, {2}))), 0, {0}))".format(
                    template_paused.format("$volume"),
                    template_paused.format(simpson_vwap),
                    template_paused.format("$low"),
                    template_paused.format("$high"),
                )
            )
        ]
        names += ["$volume_1"]

        fields += [template_paused.format("Date($close)")]
        names += ["date"]
        return fields, names


class HighFreqBacktestHandler(DataHandler):
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        freq="1min",
    ):
        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": self.get_feature_config(),
                "swap_level": False,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            freq=freq,
            data_loader=data_loader,
        )

    def get_feature_config(self):
        fields = []
        names = []

        template_if = "If(IsNull({1}), {0}, {1})"
        template_paused = "Select(Eq($paused, 0.0), {0})"
        # template_paused="{0}"
        template_fillnan = "FFillNan({0})"
        simpson_vwap = "($open + 2*$high + 2*$low + $close)/6"
        # fields += [
        #    template_fillnan.format(template_paused.format("$close")),
        # ]
        fields += [
            template_if.format(
                template_fillnan.format(template_paused.format("$close")),
                template_paused.format(simpson_vwap),
            )
        ]
        names += ["$vwap_0"]
        fields += [
            "If(IsNull({0}), 0, If(Or(Gt({1}, Mul(1.001, {3})), Lt({1}, Mul(0.999, {2}))), 0, {0}))".format(
                template_paused.format("$volume"),
                template_paused.format(simpson_vwap),
                template_paused.format("$low"),
                template_paused.format("$high"),
            )
        ]
        names += ["$volume_0"]

        return fields, names
