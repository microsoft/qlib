from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
from datetime import date, datetime
from app.db.database import get_db
from app.models.user import User
from app.schemas.data import StockDataResponse, DataResponse
from app.services.data import get_stock_data, get_stock_data_list, get_stock_codes, align_data
from app.api.deps import get_current_active_user

# 添加数据对齐请求模型
from pydantic import BaseModel

class AlignDataRequest(BaseModel):
    mode: str
    date: str

class CustomFeatureRequest(BaseModel):
    instruments: List[str]
    features: List[Dict[str, str]]
    start_date: str
    end_date: str
    freq: str = "day"

class InstrumentFilterRequest(BaseModel):
    market: Optional[str] = None
    name_filter: Optional[str] = None
    expression_filter: Optional[str] = None

router = APIRouter()

@router.get("/", response_model=DataResponse)
def read_stock_data(
    stock_code: Optional[str] = Query(None, description="Stock code to filter by"),
    market: Optional[str] = Query(None, description="Market to filter by"),
    start_date: Optional[date] = Query(None, description="Start date for filtering"),
    end_date: Optional[date] = Query(None, description="End date for filtering"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(100, ge=1, le=1000, description="Items per page"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    skip = (page - 1) * per_page
    data, total = get_stock_data_list(
        db=db,
        stock_code=stock_code,
        start_date=start_date,
        end_date=end_date,
        skip=skip,
        limit=per_page
    )
    
    return {
        "data": data,
        "total": total,
        "page": page,
        "per_page": per_page
    }

@router.get("/stock-codes", response_model=List[str])
def read_stock_codes(current_user: User = Depends(get_current_active_user)):
    # Import here to avoid circular imports
    from app.services.qlib_service import qlib_service
    # Get all available stock codes directly from QLib
    stock_codes = qlib_service.get_instruments()
    return stock_codes

@router.get("/calendar")
def get_calendar(
    start_time: str = Query(..., description="Start time in format YYYY-MM-DD"),
    end_time: str = Query(..., description="End time in format YYYY-MM-DD"),
    freq: str = Query("day", description="Frequency, default is 'day'"),
    current_user: User = Depends(get_current_active_user)
):
    # Import here to avoid circular imports
    from app.services.qlib_service import qlib_service
    
    try:
        # Initialize QLib if not already initialized
        if not qlib_service.is_initialized():
            qlib_service.init_qlib()
        
        # Get calendar from QLib
        from qlib.data import D
        calendar = D.calendar(start_time=start_time, end_time=end_time, freq=freq)
        
        # Convert to string format
        dates = [str(date) for date in calendar]
        
        return {
            "dates": dates,
            "start_date": start_time,
            "end_date": end_time,
            "freq": freq
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get calendar: {str(e)}")

@router.get("/instruments", response_model=List[str])
def get_instruments(
    market: Optional[str] = Query("all", description="Market name"),
    name_filter: Optional[str] = Query(None, description="Name filter regex"),
    expression_filter: Optional[str] = Query(None, description="Expression filter"),
    current_user: User = Depends(get_current_active_user)
):
    # Import here to avoid circular imports
    from app.services.qlib_service import qlib_service
    
    try:
        # Get instruments with filters
        instruments = qlib_service.get_instruments(market=market)
        
        # Apply additional filters if provided
        if name_filter:
            import re
            pattern = re.compile(name_filter)
            instruments = [inst for inst in instruments if pattern.match(inst)]
        
        return instruments
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get instruments: {str(e)}")

@router.post("/features")
def calculate_features(
    request: CustomFeatureRequest,
    current_user: User = Depends(get_current_active_user)
):
    # Import here to avoid circular imports
    from app.services.qlib_service import qlib_service
    
    try:
        # Initialize QLib if not already initialized
        if not qlib_service.is_initialized():
            qlib_service.init_qlib()
        
        # Get features from QLib
        from qlib.data import D
        
        # Extract feature expressions
        feature_exprs = [f["expression"] for f in request.features]
        
        # Calculate features
        data = D.features(
            instruments=request.instruments,
            fields=feature_exprs,
            start_time=request.start_date,
            end_time=request.end_date,
            freq=request.freq
        )
        
        # Convert to dict format
        result = {
            "data": data.to_dict(),
            "features": feature_exprs
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate features: {str(e)}")

@router.get("/{data_id}", response_model=StockDataResponse)
def read_stock_data_detail(
    data_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    db_data = get_stock_data(db=db, data_id=data_id)
    if db_data is None:
        raise HTTPException(status_code=404, detail="Stock data not found")
    return db_data

@router.post("/align")
def align_stock_data(
    align_request: AlignDataRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    try:
        result = align_data(
            mode=align_request.mode,
            date=align_request.date,
            db=db
        )
        return {"message": "数据对齐操作已执行", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"数据对齐失败: {str(e)}")
