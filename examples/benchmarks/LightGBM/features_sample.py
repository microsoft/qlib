import datetime
import pandas as pd

from qlib.data.inst_processor import InstProcessor


class Resample1minProcessor(InstProcessor):
    """This processor tries to resample the data. It will reasmple the data from 1min freq to day freq by selecting a specific miniute"""

    def __init__(self, hour: int, minute: int, **kwargs):
        self.hour = hour
        self.minute = minute

    def __call__(self, df: pd.DataFrame, *args, **kwargs):
        df.index = pd.to_datetime(df.index)
        df = df.loc[df.index.time == datetime.time(self.hour, self.minute)]
        df.index = df.index.normalize()
        return df
