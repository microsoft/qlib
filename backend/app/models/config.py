from sqlalchemy import Column, Integer, String, Text, DateTime, Enum
from sqlalchemy.sql import func
from app.db.database import Base
import enum

class ConfigType(str, enum.Enum):
    EXPERIMENT_TEMPLATE = "experiment_template"
    NORMAL = "normal"

class Config(Base):
    __tablename__ = "configs"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), index=True, nullable=False)
    description = Column(Text, nullable=True)
    content = Column(Text, nullable=False)  # YAML配置内容
    type = Column(Enum(ConfigType), default=ConfigType.NORMAL, nullable=False)  # 配置类型
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
