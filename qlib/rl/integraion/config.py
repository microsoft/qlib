from pathlib import Path
from typing import Union, Tuple, List, Dict, Optional

from qlib.config import configclass


@configclass
class ExchangeConfig:
    """To config a exchange object."""
    limit_threshold: Union[float, Tuple[str, str]]
    deal_price: Union[str, Tuple[str, str]]
    volume_threshold: Union[float, Dict[str, Tuple[str, str]]]
    open_cost: float = 0.0005
    close_cost: float = 0.0015
    min_cost: float = 5.
    trade_unit: Optional[float] = 100.
    cash_limit: Optional[Union[Path, float]] = None
    generate_report: bool = False


@configclass
class QlibConfig:
    """Config basic info when initalizing qlib. Could be dirty for the first iteration."""
    provider_uri_day: Path
    provider_uri_1min: Path
    feature_root_dir: Path
    feature_columns_today: List[str]
    feature_columns_yesterday: List[str]

