import datetime
import pandas as pd


def resample_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.droplevel(level="instrument")
    df = df.loc[df.index.time == datetime.time(13, 1)]
    df.index = df.index.normalize()
    return df
