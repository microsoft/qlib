from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class ExperimentBase(BaseModel):
    name: str
    description: Optional[str] = None
    config: Dict[str, Any]

class ExperimentCreate(ExperimentBase):
    pass

class ExperimentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    status: Optional[str] = None

class ExperimentResponse(ExperimentBase):
    id: int
    status: str
    created_at: datetime
    updated_at: datetime
    performance: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    class Config:
        from_attributes = True
