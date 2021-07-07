from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from qlib.contrib.data.handler import check_transform_proc


class HighFreqHandler(DataHandlerLP):
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        infer_processors=[],
        learn_processors=[],
        fit_start_time=None,
        fit_end_time=None,
        drop_raw=True,
    ):

        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": self.get_feature_config(),
                "swap_level": False,
                "freq": "1min",
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            drop_raw=drop_raw,
        )

    def get_feature_config(self):
        fields = []
        names = []

        template_if = "If(IsNull({1}), {0}, {1})"
        template_paused = "Select(Or(IsNull($paused), Eq($paused, 0.0)), {0})"
        template_fillnan = "BFillNan(FFillNan({0}))"
        # Because there is no vwap field in the yahoo data, a method similar to Simpson integration is used to approximate vwap
        simpson_vwap = "($open + 2*$high + 2*$low + $close)/6"

        def get_normalized_price_feature(price_field, shift=0):
            """Get normalized price feature ops"""
            if shift == 0:
                template_norm = "Cut({0}/Ref(DayLast({1}), 240), 240, None)"
            else:
                template_norm = "Cut(Ref({0}, " + str(shift) + ")/Ref(DayLast({1}), 240), 240, None)"

            feature_ops = template_norm.format(
                template_if.format(
                    template_fillnan.format(template_paused.format("$close")),
                    template_paused.format(price_field),
                ),
                template_fillnan.format(template_paused.format("$close")),
            )
            return feature_ops

        fields += [get_normalized_price_feature("$open", 0)]
        fields += [get_normalized_price_feature("$high", 0)]
        fields += [get_normalized_price_feature("$low", 0)]
        fields += [get_normalized_price_feature("$close", 0)]
        fields += [get_normalized_price_feature(simpson_vwap, 0)]
        names += ["$open", "$high", "$low", "$close", "$vwap"]

        fields += [get_normalized_price_feature("$open", 240)]
        fields += [get_normalized_price_feature("$high", 240)]
        fields += [get_normalized_price_feature("$low", 240)]
        fields += [get_normalized_price_feature("$close", 240)]
        fields += [get_normalized_price_feature(simpson_vwap, 240)]
        names += ["$open_1", "$high_1", "$low_1", "$close_1", "$vwap_1"]

        fields += [
            "Cut({0}/Ref(DayLast(Mean({0}, 7200)), 240), 240, None)".format(
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
            "Cut(Ref({0}, 240)/Ref(DayLast(Mean({0}, 7200)), 240), 240, None)".format(
                "If(IsNull({0}), 0, If(Or(Gt({1}, Mul(1.001, {3})), Lt({1}, Mul(0.999, {2}))), 0, {0}))".format(
                    template_paused.format("$volume"),
                    template_paused.format(simpson_vwap),
                    template_paused.format("$low"),
                    template_paused.format("$high"),
                )
            )
        ]
        names += ["$volume_1"]

        fields += ["Cut({0}, 240, None)".format(template_paused.format("Date($close)"))]
        names += ["date"]
        return fields, names


class HighFreqBacktestHandler(DataHandler):
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
    ):
        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": self.get_feature_config(),
                "swap_level": False,
                "freq": "1min",
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
        )

    def get_feature_config(self):
        fields = []
        names = []

        template_if = "If(IsNull({1}), {0}, {1})"
        template_paused = "Select(Or(IsNull($paused), Eq($paused, 0.0)), {0})"
        template_fillnan = "BFillNan(FFillNan({0}))"
        # Because there is no vwap field in the yahoo data, a method similar to Simpson integration is used to approximate vwap
        simpson_vwap = "($open + 2*$high + 2*$low + $close)/6"
        fields += [
            "Cut({0}, 240, None)".format(template_fillnan.format(template_paused.format("$close"))),
        ]
        names += ["$close0"]
        fields += [
            "Cut({0}, 240, None)".format(
                template_if.format(
                    template_fillnan.format(template_paused.format("$close")),
                    template_paused.format(simpson_vwap),
                )
            )
        ]
        names += ["$vwap0"]
        fields += [
            "Cut(If(IsNull({0}), 0, If(Or(Gt({1}, Mul(1.001, {3})), Lt({1}, Mul(0.999, {2}))), 0, {0})), 240, None)".format(
                template_paused.format("$volume"),
                template_paused.format(simpson_vwap),
                template_paused.format("$low"),
                template_paused.format("$high"),
            )
        ]
        names += ["$volume0"]

        return fields, names
