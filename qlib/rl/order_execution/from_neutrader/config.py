from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union


@dataclass
class RuntimeConfig:
    seed: int = 42
    output_dir: Optional[Path] = None
    checkpoint_dir: Optional[Path] = None
    tb_log_dir: Optional[Path] = None
    debug: bool = False
    use_cuda: bool = True


@dataclass
class ExchangeConfig:
    limit_threshold: Union[float, Tuple[str, str]]
    deal_price: Union[str, Tuple[str, str]]
    volume_threshold: dict
    open_cost: float = 0.0005
    close_cost: float = 0.0015
    min_cost: float = 5.
    trade_unit: Optional[float] = 100.
    cash_limit: Optional[Union[Path, float]] = None
    generate_report: bool = False
