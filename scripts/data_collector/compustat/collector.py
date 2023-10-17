# Copyright (c) Panoramic Hills Capital.

import sys
from pathlib import Path
from typing import Any, Optional

import fire
from qlib.data.cache import H
from qlib.log import TimeInspector
from panoramic.common.db.postgres import COMPUSTAT_DB as COMPUSTAT
from panoramic.common.db.postgres import ManagedSession, engine
from panoramic.common.model.compustat import IdxIndex

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.index import IndexBase


class CompustatIndex(IndexBase):
    TABLE_NAME = "idx_index"

    def __init__(
        self,
        gvkeyx: str,
        qlib_dir: Optional[str | Path] = None,
        freq: str = "day",
        request_retry: int = 5,
        retry_sleep: int = 3,
    ):
        self.db_obj = self._db_init(gvkeyx)
        super().__init__(self.db_obj.conm, qlib_dir, freq, request_retry, retry_sleep)

    def _db_init(self, gvkeyx):
        flag = f"{CompustatIndex.TABLE_NAME}_{gvkeyx}"
        if flag in H["x"]:
            return H["x"][flag]
        else:
            with ManagedSession(db=COMPUSTAT) as session, TimeInspector.logt("Pulling IdxIndex from DB"):
                # search IdxIndex by similar index_name
                obj = session.query(IdxIndex).filter(IdxIndex.gvkeyx == gvkeyx).first()  # type: ignore
                if not obj:
                    raise ValueError(f"Index {gvkeyx} not found in {self.TABLE_NAME}")
                session.expunge_all()
                H["x"][flag] = obj
                return obj

    def __getattr__(self, __name: str) -> Any:
        if hasattr(self.db_obj, __name):
            return getattr(self.db_obj, __name)
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {__name}")
