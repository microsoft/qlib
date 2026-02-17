from qlib.data.dataset.loader import QlibDataLoader


class Alpha360DL(QlibDataLoader):
    """Dataloader to get Alpha360"""

    def __init__(self, config=None, **kwargs):
        _config = {
            "feature": self.get_feature_config(),
        }
        if config is not None:
            _config.update(config)
        super().__init__(config=_config, **kwargs)

    @staticmethod
    def get_feature_config():
        # NOTE:
        # Alpha360 tries to provide a dataset with original price data
        # the original price data includes the prices and volume in the last 60 days.
        # To make it easier to learn models from this dataset, all the prices and volume
        # are normalized by the latest price and volume data ( dividing by $close, $volume)
        # So the latest normalized $close will be 1 (with name CLOSE0), the latest normalized $volume will be 1 (with name VOLUME0)
        # If further normalization are executed (e.g. centralization),  CLOSE0 and VOLUME0 will be 0.
        fields = []
        names = []

        for i in range(59, 0, -1):
            fields += [f"Ref($close, {i})/$close"]
            names += [f"CLOSE{i}"]
        fields += ["$close/$close"]
        names += ["CLOSE0"]
        for i in range(59, 0, -1):
            fields += [f"Ref($open, {i})/$close"]
            names += [f"OPEN{i}"]
        fields += ["$open/$close"]
        names += ["OPEN0"]
        for i in range(59, 0, -1):
            fields += [f"Ref($high, {i})/$close"]
            names += [f"HIGH{i}"]
        fields += ["$high/$close"]
        names += ["HIGH0"]
        for i in range(59, 0, -1):
            fields += [f"Ref($low, {i})/$close"]
            names += [f"LOW{i}"]
        fields += ["$low/$close"]
        names += ["LOW0"]
        for i in range(59, 0, -1):
            fields += [f"Ref($vwap, {i})/$close"]
            names += [f"VWAP{i}"]
        fields += ["$vwap/$close"]
        names += ["VWAP0"]
        for i in range(59, 0, -1):
            fields += [f"Ref($volume, {i})/($volume+1e-12)"]
            names += [f"VOLUME{i}"]
        fields += ["$volume/($volume+1e-12)"]
        names += ["VOLUME0"]

        return fields, names


class Alpha158DL(QlibDataLoader):
    """Dataloader to get Alpha158"""

    def __init__(self, config=None, **kwargs):
        _config = {
            "feature": self.get_feature_config(),
        }
        if config is not None:
            _config.update(config)
        super().__init__(config=_config, **kwargs)

    @staticmethod
    def get_feature_config(
        config=None
    ):
        if config is None:
            config = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
        """create factors from config

        config = {
            'kbar': {}, # whether to use some hard-code kbar features
            'price': { # whether to use raw price features
                'windows': [0, 1, 2, 3, 4], # use price at n days ago
                'feature': ['OPEN', 'HIGH', 'LOW'] # which price field to use
            },
            'volume': { # whether to use raw volume features
                'windows': [0, 1, 2, 3, 4], # use volume at n days ago
            },
            'rolling': { # whether to use rolling operator based features
                'windows': [5, 10, 20, 30, 60], # rolling windows size
                'include': ['ROC', 'MA', 'STD'], # rolling operator to use
                #if include is None we will use default operators
                'exclude': ['RANK'], # rolling operator not to use
            }
        }
        """
        fields = []
        names = []
        if "kbar" in config:
            fields += [
                "($close-$open)/$open",
                "($high-$low)/$open",
                "($close-$open)/($high-$low+1e-12)",
                "($high-Greater($open, $close))/$open",
                "($high-Greater($open, $close))/($high-$low+1e-12)",
                "(Less($open, $close)-$low)/$open",
                "(Less($open, $close)-$low)/($high-$low+1e-12)",
                "(2*$close-$high-$low)/$open",
                "(2*$close-$high-$low)/($high-$low+1e-12)",
            ]
            names += [
                "KMID",
                "KLEN",
                "KMID2",
                "KUP",
                "KUP2",
                "KLOW",
                "KLOW2",
                "KSFT",
                "KSFT2",
            ]
        if "price" in config:
            windows = config["price"].get("windows", range(5))
            feature = config["price"].get("feature", ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"])
            for field in feature:
                field = field.lower()
                fields += [f"Ref(${field}, {d})/$close" if d != 0 else f"${field}/$close" for d in windows]
                names += [field.upper() + str(d) for d in windows]
        if "volume" in config:
            windows = config["volume"].get("windows", range(5))
            fields += [f"Ref($volume, {d})/($volume+1e-12)" if d != 0 else "$volume/($volume+1e-12)" for d in windows]
            names += ["VOLUME" + str(d) for d in windows]
        if "rolling" in config:
            windows = config["rolling"].get("windows", [5, 10, 20, 30, 60])
            include = config["rolling"].get("include", None)
            exclude = config["rolling"].get("exclude", [])
            # `exclude` in dataset config unnecessary filed
            # `include` in dataset config necessary field

            def use(x):
                return x not in exclude and (include is None or x in include)

            # Some factor ref: https://guorn.com/static/upload/file/3/134065454575605.pdf
            if use("ROC"):
                # https://www.investopedia.com/terms/r/rateofchange.asp
                # Rate of change, the price change in the past d days, divided by latest close price to remove unit
                fields += [f"Ref($close, {d})/$close" for d in windows]
                names += [f"ROC{d}" for d in windows]
            if use("MA"):
                # https://www.investopedia.com/ask/answers/071414/whats-difference-between-moving-average-and-weighted-moving-average.asp
                # Simple Moving Average, the simple moving average in the past d days, divided by latest close price to remove unit
                fields += [f"Mean($close, {d})/$close" for d in windows]
                names += [f"MA{d}" for d in windows]
            if use("STD"):
                # The standard diviation of close price for the past d days, divided by latest close price to remove unit
                fields += [f"Std($close, {d})/$close" for d in windows]
                names += [f"STD{d}" for d in windows]
            if use("BETA"):
                # The rate of close price change in the past d days, divided by latest close price to remove unit
                # For example, price increase 10 dollar per day in the past d days, then Slope will be 10.
                fields += [f"Slope($close, {d})/$close" for d in windows]
                names += [f"BETA{d}" for d in windows]
            if use("RSQR"):
                # The R-sqaure value of linear regression for the past d days, represent the trend linear
                fields += [f"Rsquare($close, {d})" for d in windows]
                names += [f"RSQR{d}" for d in windows]
            if use("RESI"):
                # The redisdual for linear regression for the past d days, represent the trend linearity for past d days.
                fields += [f"Resi($close, {d})/$close" for d in windows]
                names += [f"RESI{d}" for d in windows]
            if use("MAX"):
                # The max price for past d days, divided by latest close price to remove unit
                fields += [f"Max($high, {d})/$close" for d in windows]
                names += [f"MAX{d}" for d in windows]
            if use("LOW"):
                # The low price for past d days, divided by latest close price to remove unit
                fields += [f"Min($low, {d})/$close" for d in windows]
                names += [f"MIN{d}" for d in windows]
            if use("QTLU"):
                # The 80% quantile of past d day's close price, divided by latest close price to remove unit
                # Used with MIN and MAX
                fields += [f"Quantile($close, {d}, 0.8)/$close" for d in windows]
                names += [f"QTLU{d}" for d in windows]
            if use("QTLD"):
                # The 20% quantile of past d day's close price, divided by latest close price to remove unit
                fields += [f"Quantile($close, {d}, 0.2)/$close" for d in windows]
                names += [f"QTLD{d}" for d in windows]
            if use("RANK"):
                # Get the percentile of current close price in past d day's close price.
                # Represent the current price level comparing to past N days, add additional information to moving average.
                fields += [f"Rank($close, {d})" for d in windows]
                names += [f"RANK{d}" for d in windows]
            if use("RSV"):
                # Represent the price position between upper and lower resistent price for past d days.
                fields += [f"($close-Min($low, {d}))/(Max($high, {d})-Min($low, {d})+1e-12)" for d in windows]
                names += [f"RSV{d}" for d in windows]
            if use("IMAX"):
                # The number of days between current date and previous highest price date.
                # Part of Aroon Indicator https://www.investopedia.com/terms/a/aroon.asp
                # The indicator measures the time between highs and the time between lows over a time period.
                # The idea is that strong uptrends will regularly see new highs, and strong downtrends will regularly see new lows.
                fields += [f"IdxMax($high, {d})/{d}" for d in windows]
                names += [f"IMAX{d}" for d in windows]
            if use("IMIN"):
                # The number of days between current date and previous lowest price date.
                # Part of Aroon Indicator https://www.investopedia.com/terms/a/aroon.asp
                # The indicator measures the time between highs and the time between lows over a time period.
                # The idea is that strong uptrends will regularly see new highs, and strong downtrends will regularly see new lows.
                fields += [f"IdxMin($low, {d})/{d}" for d in windows]
                names += [f"IMIN{d}" for d in windows]
            if use("IMXD"):
                # The time period between previous lowest-price date occur after highest price date.
                # Large value suggest downward momemtum.
                fields += [f"(IdxMax($high, {d})-IdxMin($low, {d}))/{d}" for d in windows]
                names += [f"IMXD{d}" for d in windows]
            if use("CORR"):
                # The correlation between absolute close price and log scaled trading volume
                fields += [f"Corr($close, Log($volume+1), {d})" for d in windows]
                names += [f"CORR{d}" for d in windows]
            if use("CORD"):
                # The correlation between price change ratio and volume change ratio
                fields += [f"Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), {d})" for d in windows]
                names += [f"CORD{d}" for d in windows]
            if use("CNTP"):
                # The percentage of days in past d days that price go up.
                fields += [f"Mean($close>Ref($close, 1), {d})" for d in windows]
                names += [f"CNTP{d}" for d in windows]
            if use("CNTN"):
                # The percentage of days in past d days that price go down.
                fields += [f"Mean($close<Ref($close, 1), {d})" for d in windows]
                names += [f"CNTN{d}" for d in windows]
            if use("CNTD"):
                # The diff between past up day and past down day
                fields += [f"Mean($close>Ref($close, 1), {d})-Mean($close<Ref($close, 1), {d})" for d in windows]
                names += [f"CNTD{d}" for d in windows]
            if use("SUMP"):
                # The total gain / the absolute total price changed
                # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
                fields += [
                    f"Sum(Greater($close-Ref($close, 1), 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)"
                    for d in windows
                ]
                names += [f"SUMP{d}" for d in windows]
            if use("SUMN"):
                # The total lose / the absolute total price changed
                # Can be derived from SUMP by SUMN = 1 - SUMP
                # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
                fields += [
                    f"Sum(Greater(Ref($close, 1)-$close, 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)"
                    for d in windows
                ]
                names += [f"SUMN{d}" for d in windows]
            if use("SUMD"):
                # The diff ratio between total gain and total lose
                # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
                fields += [
                    f"(Sum(Greater($close-Ref($close, 1), 0), {d})-Sum(Greater(Ref($close, 1)-$close, 0), {d}))"
                    f"/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)"
                    for d in windows
                ]
                names += [f"SUMD{d}" for d in windows]
            if use("VMA"):
                # Simple Volume Moving average: https://www.barchart.com/education/technical-indicators/volume_moving_average
                fields += [f"Mean($volume, {d})/($volume+1e-12)" for d in windows]
                names += [f"VMA{d}" for d in windows]
            if use("VSTD"):
                # The standard deviation for volume in past d days.
                fields += [f"Std($volume, {d})/($volume+1e-12)" for d in windows]
                names += [f"VSTD{d}" for d in windows]
            if use("WVMA"):
                # The volume weighted price change volatility
                fields += [
                    f"Std(Abs($close/Ref($close, 1)-1)*$volume, {d})/(Mean(Abs($close/Ref($close, 1)-1)*$volume, {d})+1e-12)"
                    for d in windows
                ]
                names += [f"WVMA{d}" for d in windows]
            if use("VSUMP"):
                # The total volume increase / the absolute total volume changed
                fields += [
                    f"Sum(Greater($volume-Ref($volume, 1), 0), {d})/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)"
                    for d in windows
                ]
                names += [f"VSUMP{d}" for d in windows]
            if use("VSUMN"):
                # The total volume increase / the absolute total volume changed
                # Can be derived from VSUMP by VSUMN = 1 - VSUMP
                fields += [
                    f"Sum(Greater(Ref($volume, 1)-$volume, 0), {d})/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)"
                    for d in windows
                ]
                names += [f"VSUMN{d}" for d in windows]
            if use("VSUMD"):
                # The diff ratio between total volume increase and total volume decrease
                # RSI indicator for volume
                fields += [
                    f"(Sum(Greater($volume-Ref($volume, 1), 0), {d})-Sum(Greater(Ref($volume, 1)-$volume, 0), {d}))"
                    f"/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)"
                    for d in windows
                ]
                names += [f"VSUMD{d}" for d in windows]

        return fields, names
