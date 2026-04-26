import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import h5py
import pandas as pd
import fire
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


RQALPHA_TO_QLIB_EXCHANGE = {
    "XSHE": "SZ",
    "XSHG": "SH",
}


def rqalpha_code_to_qlib_code(order_book_id: str) -> str:
    code, exchange = order_book_id.split(".")
    try:
        return f"{RQALPHA_TO_QLIB_EXCHANGE[exchange]}{code}"
    except KeyError as exc:
        raise ValueError(f"unsupported exchange in order_book_id: {order_book_id}") from exc


def rqalpha_datetime_to_timestamp(value: int) -> pd.Timestamp:
    return pd.to_datetime(str(int(value)), format="%Y%m%d%H%M%S")


class RQAlphaBundleConverter:
    STOCKS_FILE = "stocks.h5"
    INDEXES_FILE = "indexes.h5"
    EX_CUM_FACTOR_FILE = "ex_cum_factor.h5"
    DEFAULT_FIELDS = ("open", "close", "high", "low", "volume", "vwap", "factor")
    PRICE_FIELDS = {"open", "close", "high", "low", "vwap", "prev_close", "limit_up", "limit_down"}

    def __init__(self, bundle_path: str):
        self.bundle_path = Path(bundle_path).expanduser().resolve()
        if not self.bundle_path.exists():
            raise FileNotFoundError(f"bundle path does not exist: {self.bundle_path}")

    def _bundle_file(self, instrument_type: str) -> Path:
        if instrument_type == "stocks":
            return self.bundle_path / self.STOCKS_FILE
        if instrument_type == "indexes":
            return self.bundle_path / self.INDEXES_FILE
        raise ValueError(f"unsupported instrument_type: {instrument_type}")

    def _factor_rows(self, order_book_id: str):
        factor_file = self.bundle_path / self.EX_CUM_FACTOR_FILE
        if not factor_file.exists():
            return None
        with h5py.File(factor_file, "r") as factors:
            if order_book_id not in factors:
                return None
            return factors[order_book_id][:]

    @staticmethod
    def _align_factor(df: pd.DataFrame, factor_rows) -> pd.Series:
        if factor_rows is None or len(factor_rows) == 0:
            return pd.Series(1.0, index=df.index)

        factor_df = pd.DataFrame.from_records(factor_rows)
        if factor_df.empty or "ex_cum_factor" not in factor_df.columns:
            return pd.Series(1.0, index=df.index)

        factor_df = factor_df.copy()
        factor_df["date"] = factor_df["start_date"].map(
            lambda value: pd.Timestamp("1900-01-01")
            if int(value) == 0
            else rqalpha_datetime_to_timestamp(value)
        )
        factor_df = factor_df.loc[:, ["date", "ex_cum_factor"]].sort_values("date")
        aligned = pd.merge_asof(
            df.loc[:, ["date"]].sort_values("date"),
            factor_df,
            on="date",
            direction="backward",
        )
        return aligned["ex_cum_factor"].fillna(1.0).set_axis(df.sort_values("date").index).reindex(df.index)

    @classmethod
    def _normalize_dataset(cls, rows, fields: Iterable[str], order_book_id: str, factor_rows=None) -> pd.DataFrame:
        df = pd.DataFrame.from_records(rows)
        if df.empty:
            return df
        fields = tuple(fields)
        if "vwap" in fields and "vwap" not in df.columns:
            if "total_turnover" not in df.columns or "volume" not in df.columns:
                raise ValueError(f"`vwap` requires `total_turnover` and `volume` in source data: {order_book_id}")
            df["vwap"] = (df["total_turnover"] / df["volume"]).where(df["volume"] > 0, df["close"])
        df["date"] = df["datetime"].map(rqalpha_datetime_to_timestamp)
        df["symbol"] = rqalpha_code_to_qlib_code(order_book_id)

        factor = cls._align_factor(df, factor_rows)
        if "factor" in fields:
            df["factor"] = factor
        for field in cls.PRICE_FIELDS.intersection(fields).intersection(df.columns):
            df[field] = df[field] * factor
        if "volume" in fields and "volume" in df.columns:
            df["volume"] = (df["volume"] / factor).where(factor > 0, df["volume"])

        columns = ["date", "symbol", *fields]
        return df.loc[:, columns].sort_values("date")

    def export_csv(
        self,
        output_dir: str,
        instrument_type: str = "stocks",
        include_fields: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> str:
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        fields = self.DEFAULT_FIELDS if include_fields is None else tuple(
            field.strip() for field in include_fields.split(",") if field.strip()
        )
        bundle_file = self._bundle_file(instrument_type)

        with h5py.File(bundle_file, "r") as bundle:
            order_book_ids: List[str] = sorted(bundle.keys())
            if limit is not None:
                order_book_ids = order_book_ids[: int(limit)]

            for order_book_id in tqdm(order_book_ids, desc=f"export {instrument_type}"):
                df = self._normalize_dataset(bundle[order_book_id][:], fields, order_book_id, self._factor_rows(order_book_id))
                if df.empty:
                    continue
                csv_path = output_path / f"{rqalpha_code_to_qlib_code(order_book_id).lower()}.csv"
                df.to_csv(csv_path, index=False)

        return str(output_path)

    def export_and_dump_qlib(
        self,
        csv_dir: str,
        qlib_dir: str,
        instrument_type: str = "stocks",
        include_fields: Optional[str] = None,
        limit: Optional[int] = None,
        max_workers: int = 16,
    ) -> Dict[str, str]:
        from dump_bin import DumpDataAll

        csv_path = self.export_csv(
            output_dir=csv_dir,
            instrument_type=instrument_type,
            include_fields=include_fields,
            limit=limit,
        )
        fields = ",".join(self.DEFAULT_FIELDS if include_fields is None else [
            field.strip() for field in include_fields.split(",") if field.strip()
        ])
        DumpDataAll(
            data_path=csv_path,
            qlib_dir=qlib_dir,
            include_fields=fields,
            max_workers=max_workers,
        ).dump()
        return {"csv_dir": csv_path, "qlib_dir": str(Path(qlib_dir).expanduser().resolve())}


def export_csv(bundle_path: str, output_dir: str, instrument_type: str = "stocks", include_fields: Optional[str] = None, limit: Optional[int] = None) -> str:
    return RQAlphaBundleConverter(bundle_path).export_csv(
        output_dir=output_dir,
        instrument_type=instrument_type,
        include_fields=include_fields,
        limit=limit,
    )


def export_and_dump_qlib(
    bundle_path: str,
    csv_dir: str,
    qlib_dir: str,
    instrument_type: str = "stocks",
    include_fields: Optional[str] = None,
    limit: Optional[int] = None,
    max_workers: int = 16,
) -> Dict[str, str]:
    return RQAlphaBundleConverter(bundle_path).export_and_dump_qlib(
        csv_dir=csv_dir,
        qlib_dir=qlib_dir,
        instrument_type=instrument_type,
        include_fields=include_fields,
        limit=limit,
        max_workers=max_workers,
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "export_csv": export_csv,
            "export_and_dump_qlib": export_and_dump_qlib,
        }
    )
