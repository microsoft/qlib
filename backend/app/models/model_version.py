from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from app.db.database import Base

class ModelVersion(Base):
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    version = Column(Integer, nullable=False)  # 模型版本号
    name = Column(String(100), index=True, nullable=False)
    metrics = Column(JSON, nullable=True)  # 模型指标
    path = Column(String(255), nullable=False)  # 模型存储路径
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    performance = Column(JSON, nullable=True)  # 收益数据
