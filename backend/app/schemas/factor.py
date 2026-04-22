from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

# Factor Group schemas
class FactorGroupBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    status: Optional[str] = "active"
    factor_count: Optional[int] = 0

class FactorGroupCreate(FactorGroupBase):
    pass

class FactorGroupUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    status: Optional[str] = None
    factor_count: Optional[int] = None

class FactorGroupResponse(FactorGroupBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# Factor schemas
class FactorBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    formula: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1, max_length=50)
    status: Optional[str] = "active"
    group_id: Optional[int] = None

class FactorCreate(FactorBase):
    pass

class FactorUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    formula: Optional[str] = Field(None, min_length=1)
    type: Optional[str] = Field(None, min_length=1, max_length=50)
    status: Optional[str] = None
    group_id: Optional[int] = None

class FactorResponse(FactorBase):
    id: int
    created_at: datetime
    updated_at: datetime
    group: Optional[FactorGroupResponse] = None

    class Config:
        from_attributes = True

# Factor Group with factors schema
class FactorGroupWithFactors(FactorGroupResponse):
    factors: List[FactorResponse] = []

    class Config:
        from_attributes = True
