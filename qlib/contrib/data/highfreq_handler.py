from qlib.data.dataset.handler import DataHandler, DataHandlerLP

from .handler import check_transform_proc

EPSILON = 1e-4


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
        template_paused = "Select(Gt($paused_num, 1.001), {0})"

        def get_normalized_price_feature(price_field, shift=0):
            # norm with the close price of 237th minute of yesterday.
            if shift == 0:
                template_norm = "{0}/DayLast(Ref({1}, 243))"
            else:
                template_norm = "Ref({0}, " + str(shift) + ")/DayLast(Ref({1}, 243))"

            template_fillnan = "FFillNan({0})"
            # calculate -> ffill -> remove paused
            feature_ops = template_paused.format(
                template_fillnan.format(
                    template_norm.format(template_if.format("$close", price_field), template_fillnan.format("$close"))
                )
            )
            return feature_ops

        fields += [get_normalized_price_feature("$open", 0)]
        fields += [get_normalized_price_feature("$high", 0)]
        fields += [get_normalized_price_feature("$low", 0)]
        fields += [get_normalized_price_feature("$close", 0)]
        fields += [get_normalized_price_feature("$vwap", 0)]
        names += ["$open", "$high", "$low", "$close", "$vwap"]

        fields += [get_normalized_price_feature("$open", 240)]
        fields += [get_normalized_price_feature("$high", 240)]
        fields += [get_normalized_price_feature("$low", 240)]
        fields += [get_normalized_price_feature("$close", 240)]
        fields += [get_normalized_price_feature("$vwap", 240)]
        names += ["$open_1", "$high_1", "$low_1", "$close_1", "$vwap_1"]

        # calculate and fill nan with 0
        template_gzero = "If(Ge({0}, 0), {0}, 0)"
        fields += [
            template_gzero.format(
                template_paused.format(
                    "If(IsNull({0}), 0, {0})".format("{0}/Ref(DayLast(Mean({0}, 7200)), 240)".format("$volume"))
                )
            )
        ]
        names += ["$volume"]

        fields += [
            template_gzero.format(
                template_paused.format(
                    "If(IsNull({0}), 0, {0})".format(
                        "Ref({0}, 240)/Ref(DayLast(Mean({0}, 7200)), 240)".format("$volume")
                    )
                )
            )
        ]
        names += ["$volume_1"]

        return fields, names


class HighFreqGeneralHandler(DataHandlerLP):
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
        day_length=240,
        freq="1min",
        columns=["$open", "$high", "$low", "$close", "$vwap"],
        inst_processors=None,
    ):
        self.day_length = day_length
        self.columns = columns

        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": self.get_feature_config(),
                "swap_level": False,
                "freq": freq,
                "inst_processors": inst_processors,
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
        template_paused = f"Cut({{0}}, {self.day_length * 2}, None)"

        def get_normalized_price_feature(price_field, shift=0):
            # norm with the close price of 237th minute of yesterday.
            if shift == 0:
                template_norm = f"{{0}}/DayLast(Ref({{1}}, {self.day_length * 2}))"
            else:
                template_norm = f"Ref({{0}}, " + str(shift) + f")/DayLast(Ref({{1}}, {self.day_length}))"

            template_fillnan = "FFillNan({0})"
            # calculate -> ffill -> remove paused
            feature_ops = template_paused.format(
                template_fillnan.format(
                    template_norm.format(template_if.format("$close", price_field), template_fillnan.format("$close"))
                )
            )
            return feature_ops

        for column_name in self.columns:
            fields.append(get_normalized_price_feature(column_name, 0))
            names.append(column_name)

        for column_name in self.columns:
            fields.append(get_normalized_price_feature(column_name, self.day_length))
            names.append(column_name + "_1")

        # calculate and fill nan with 0
        fields += [
            template_paused.format(
                "If(IsNull({0}), 0, {0})".format(
                    f"{{0}}/Ref(DayLast(Mean({{0}}, {self.day_length * 30})), {self.day_length})".format("$volume")
                )
            )
        ]
        names += ["$volume"]

        fields += [
            template_paused.format(
                "If(IsNull({0}), 0, {0})".format(
                    f"Ref({{0}}, {self.day_length})/Ref(DayLast(Mean({{0}}, {self.day_length * 30})), {self.day_length})".format(
                        "$volume"
                    )
                )
            )
        ]
        names += ["$volume_1"]

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
        template_paused = "Select(Gt($paused_num, 1.001), {0})"
        template_fillnan = "FFillNan({0})"
        fields += [
            template_fillnan.format(template_paused.format("$close")),
        ]
        names += ["$close0"]

        fields += [
            template_paused.format(
                template_if.format(
                    template_fillnan.format("$close"),
                    "$vwap",
                )
            )
        ]
        names += ["$vwap0"]

        fields += [template_paused.format("If(IsNull({0}), 0, {0})".format("$volume"))]
        names += ["$volume0"]

        fields += [template_paused.format("If(IsNull({0}), 0, {0})".format("$factor"))]
        names += ["$factor0"]

        return fields, names


class HighFreqGeneralBacktestHandler(DataHandler):
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        day_length=240,
        freq="1min",
        columns=["$close", "$vwap", "$volume"],
        inst_processors=None,
    ):
        self.day_length = day_length
        self.columns = set(columns)
        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": self.get_feature_config(),
                "swap_level": False,
                "freq": freq,
                "inst_processors": inst_processors,
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

        if "$close" in self.columns:
            template_paused = f"Cut({{0}}, {self.day_length * 2}, None)"
            template_fillnan = "FFillNan({0})"
            template_if = "If(IsNull({1}), {0}, {1})"
            fields += [
                template_paused.format(template_fillnan.format("$close")),
            ]
            names += ["$close0"]

        if "$vwap" in self.columns:
            fields += [
                template_paused.format(template_if.format(template_fillnan.format("$close"), "$vwap")),
            ]
            names += ["$vwap0"]

        if "$volume" in self.columns:
            fields += [template_paused.format("If(IsNull({0}), 0, {0})".format("$volume"))]
            names += ["$volume0"]

        return fields, names


class HighFreqOrderHandler(DataHandlerLP):
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        infer_processors=[],
        learn_processors=[],
        fit_start_time=None,
        fit_end_time=None,
        inst_processors=None,
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
                "inst_processors": inst_processors,
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
        template_ifinf = "If(IsInf({1}), {0}, {1})"
        template_paused = "Select(Gt($paused_num, 1.001), {0})"

        def get_normalized_price_feature(price_field, shift=0):
            # norm with the close price of 237th minute of yesterday.
            if shift == 0:
                template_norm = "{0}/DayLast(Ref({1}, 243))"
            else:
                template_norm = "Ref({0}, " + str(shift) + ")/DayLast(Ref({1}, 243))"

            template_fillnan = "FFillNan({0})"
            # calculate -> ffill -> remove paused
            feature_ops = template_paused.format(
                template_fillnan.format(
                    template_norm.format(template_if.format("$close", price_field), template_fillnan.format("$close"))
                )
            )
            return feature_ops

        def get_normalized_vwap_price_feature(price_field, shift=0):
            # norm with the close price of 237th minute of yesterday.
            if shift == 0:
                template_norm = "{0}/DayLast(Ref({1}, 243))"
            else:
                template_norm = "Ref({0}, " + str(shift) + ")/DayLast(Ref({1}, 243))"

            template_fillnan = "FFillNan({0})"
            # calculate -> ffill -> remove paused
            feature_ops = template_paused.format(
                template_fillnan.format(
                    template_norm.format(
                        template_if.format("$close", template_ifinf.format("$close", price_field)),
                        template_fillnan.format("$close"),
                    )
                )
            )
            return feature_ops

        fields += [get_normalized_price_feature("$open", 0)]
        fields += [get_normalized_price_feature("$high", 0)]
        fields += [get_normalized_price_feature("$low", 0)]
        fields += [get_normalized_price_feature("$close", 0)]
        fields += [get_normalized_vwap_price_feature("$vwap", 0)]
        names += ["$open", "$high", "$low", "$close", "$vwap"]

        fields += [get_normalized_price_feature("$open", 240)]
        fields += [get_normalized_price_feature("$high", 240)]
        fields += [get_normalized_price_feature("$low", 240)]
        fields += [get_normalized_price_feature("$close", 240)]
        fields += [get_normalized_vwap_price_feature("$vwap", 240)]
        names += ["$open_1", "$high_1", "$low_1", "$close_1", "$vwap_1"]

        fields += [get_normalized_price_feature("$bid", 0)]
        fields += [get_normalized_price_feature("$ask", 0)]
        names += ["$bid", "$ask"]

        fields += [get_normalized_price_feature("$bid", 240)]
        fields += [get_normalized_price_feature("$ask", 240)]
        names += ["$bid_1", "$ask_1"]

        # calculate and fill nan with 0

        def get_volume_feature(volume_field, shift=0):
            template_gzero = "If(Ge({0}, 0), {0}, 0)"
            if shift == 0:
                feature_ops = template_gzero.format(
                    template_paused.format(
                        "If(IsInf({0}), 0, {0})".format(
                            "If(IsNull({0}), 0, {0})".format(
                                "{0}/Ref(DayLast(Mean({0}, 7200)), 240)".format(volume_field)
                            )
                        )
                    )
                )
            else:
                feature_ops = template_gzero.format(
                    template_paused.format(
                        "If(IsInf({0}), 0, {0})".format(
                            "If(IsNull({0}), 0, {0})".format(
                                f"Ref({{0}}, {shift})/Ref(DayLast(Mean({{0}}, 7200)), 240)".format(volume_field)
                            )
                        )
                    )
                )
            return feature_ops

        fields += [get_volume_feature("$volume", 0)]
        names += ["$volume"]

        fields += [get_volume_feature("$volume", 240)]
        names += ["$volume_1"]

        fields += [get_volume_feature("$bidV", 0)]
        fields += [get_volume_feature("$bidV1", 0)]
        fields += [get_volume_feature("$bidV3", 0)]
        fields += [get_volume_feature("$bidV5", 0)]
        fields += [get_volume_feature("$askV", 0)]
        fields += [get_volume_feature("$askV1", 0)]
        fields += [get_volume_feature("$askV3", 0)]
        fields += [get_volume_feature("$askV5", 0)]
        names += ["$bidV", "$bidV1", "$bidV3", "$bidV5", "$askV", "$askV1", "$askV3", "$askV5"]

        fields += [get_volume_feature("$bidV", 240)]
        fields += [get_volume_feature("$bidV1", 240)]
        fields += [get_volume_feature("$bidV3", 240)]
        fields += [get_volume_feature("$bidV5", 240)]
        fields += [get_volume_feature("$askV", 240)]
        fields += [get_volume_feature("$askV1", 240)]
        fields += [get_volume_feature("$askV3", 240)]
        fields += [get_volume_feature("$askV5", 240)]
        names += ["$bidV_1", "$bidV1_1", "$bidV3_1", "$bidV5_1", "$askV_1", "$askV1_1", "$askV3_1", "$askV5_1"]

        return fields, names


class HighFreqBacktestOrderHandler(DataHandler):
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
        template_paused = "Select(Gt($paused_num, 1.001), {0})"
        template_fillnan = "FFillNan({0})"
        fields += [
            template_fillnan.format(template_paused.format("$close")),
        ]
        names += ["$close0"]

        fields += [
            template_paused.format(
                template_if.format(
                    template_fillnan.format("$close"),
                    "$vwap",
                )
            )
        ]
        names += ["$vwap0"]

        fields += [template_paused.format("If(IsNull({0}), 0, {0})".format("$volume"))]
        names += ["$volume0"]

        fields += [template_paused.format("If(IsNull({0}), 0, {0})".format("$bid"))]
        names += ["$bid0"]

        fields += [template_paused.format("If(IsNull({0}), 0, {0})".format("$bidV"))]
        names += ["$bidV0"]

        fields += [template_paused.format("If(IsNull({0}), 0, {0})".format("$ask"))]
        names += ["$ask0"]

        fields += [template_paused.format("If(IsNull({0}), 0, {0})".format("$askV"))]
        names += ["$askV0"]

        fields += [template_paused.format("If(IsNull({0}), 0, {0})".format("($bid + $ask) / 2"))]
        names += ["$median0"]

        fields += [template_paused.format("If(IsNull({0}), 0, {0})".format("$factor"))]
        names += ["$factor0"]

        fields += [template_paused.format("If(IsNull({0}), 0, {0})".format("$downlimitmarket"))]
        names += ["$downlimitmarket0"]

        fields += [template_paused.format("If(IsNull({0}), 0, {0})".format("$uplimitmarket"))]
        names += ["$uplimitmarket0"]

        fields += [template_paused.format("If(IsNull({0}), 0, {0})".format("$highmarket"))]
        names += ["$highmarket0"]

        fields += [template_paused.format("If(IsNull({0}), 0, {0})".format("$lowmarket"))]
        names += ["$lowmarket0"]

        return fields, names
