import pandas as pd
from typing import Union


def split_df_to_str(df: Union[pd.DataFrame, str]) -> str:
    if not isinstance(df, str):
        df = str(df)
    return "".join(df.split())
