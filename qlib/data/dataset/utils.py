from typing import Union
import pandas as pd


def get_level_index(df: pd.DataFrame, level=Union[str, int]) -> int:
    """

    get the level index of `df` given `level`

    Parameters
    ----------
    df : pd.DataFrame
        data
    level : Union[str, int]
        index level

    Returns
    -------
    int:
        The level index in the multiple index
    """
    if isinstance(level, str):
        try:
            return df.index.names.index(level)
        except (AttributeError, ValueError):
            # NOTE: If level index is not given in the data, the default level index will be ('datetime', 'instrument')
            return ('datetime', 'instrument').index(level)
    elif isinstance(level, int):
        return level
    else:
        raise NotImplementedError(f"This type of input is not supported")

