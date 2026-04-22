from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, date

class StockDataBase(BaseModel):
    stock_code: str = Field(..., min_length=1, max_length=20)
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float

class StockDataCreate(StockDataBase):
    pass

class StockDataResponse(StockDataBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class DataFilter(BaseModel):
    stock_code: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None

class DataResponse(BaseModel):
    data: List[StockDataResponse]
    total: int
    page: int
    per_page: int
