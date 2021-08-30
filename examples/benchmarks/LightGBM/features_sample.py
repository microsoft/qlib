import datetime
import pandas as pd

from qlib.data.inst_processor import InstProcessor


class ResampleProcessor(InstProcessor):
    def __init__(self, freq: str, hour: int, minute: int):
        self.freq = freq
        self.hour = hour
        self.minute = minute

    def __call__(self, df: pd.DataFrame, *args, **kwargs):
        df.index = pd.to_datetime(df.index)
        df = df.loc[df.index.time == datetime.time(self.hour, self.minute)]
        df.index = df.index.normalize()
        return df
