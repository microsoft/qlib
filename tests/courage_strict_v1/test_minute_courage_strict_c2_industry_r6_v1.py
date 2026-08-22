from __future__ import annotations

import pandas as pd

from qlib.contrib.data.courage_strict_v1.minute_courage_strict_c2_industry_r6_v1 import (
    assign_industry_strictly_before_v1,
    normalize_provider_history_v1,
)


def test_provider_normalization() -> None:
    raw = pd.DataFrame(
        {
            "证券代码": ["000001"],
            "变更日期": ["2025-04-01"],
            "分类标准编码": ["008002"],
            "分类标准": ["巨潮行业分类标准"],
            "行业编码": ["Z07010101"],
            "行业门类": ["金融"],
            "行业次类": ["银行"],
            "行业大类": ["银行"],
            "行业中类": ["银行"],
        }
    )
    result = normalize_provider_history_v1("000001.SZ", raw)
    assert result.loc[0, "instrument"] == "000001.SZ"
    assert result.loc[0, "classification_standard_code"] == "008002"


def test_assignment_is_strictly_before_and_unknown_is_explicit() -> None:
    daily = pd.DataFrame(
        {
            "instrument": ["A.SH", "A.SH", "B.SZ"],
            "session_date": pd.to_datetime(["2025-04-01", "2025-04-02", "2025-04-02"]),
        }
    )
    history = pd.DataFrame(
        {
            "instrument": ["A.SH"],
            "change_date": pd.to_datetime(["2025-04-01"]),
            "classification_standard_code": ["008002"],
            "industry_code": ["Z01010101"],
            "sector_level2_code": ["Z0101"],
            "sector_level2_name": ["能源"],
        }
    )
    result = assign_industry_strictly_before_v1(daily, history)
    first = result.loc[
        (result.instrument == "A.SH")
        & (result.session_date == pd.Timestamp("2025-04-01"))
    ].iloc[0]
    second = result.loc[
        (result.instrument == "A.SH")
        & (result.session_date == pd.Timestamp("2025-04-02"))
    ].iloc[0]
    unknown = result.loc[result.instrument == "B.SZ"].iloc[0]
    assert not first.industry_known
    assert second.industry_known
    assert not unknown.industry_known
    assert not result["same_change_date_use_authorized"].any()
    assert not result["future_fill_authorized"].any()
