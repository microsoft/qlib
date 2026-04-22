from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from app.models.config import ConfigType

class ConfigBase(BaseModel):
    name: str
    description: Optional[str] = None
    content: str
    type: Optional[ConfigType] = ConfigType.NORMAL

class ConfigCreate(ConfigBase):
    pass

class ConfigUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None
    type: Optional[ConfigType] = None

class ConfigResponse(ConfigBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True
