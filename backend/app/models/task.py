from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey
from sqlalchemy.sql import func
from app.db.database import Base

class Task(Base):
    __tablename__ = "tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    status = Column(String(20), default="pending", nullable=False)  # pending, running, completed, failed
    task_type = Column(String(20), default="train", nullable=False)  # train, predict, etc.
    priority = Column(Integer, default=0, nullable=False)
    retries = Column(Integer, default=0, nullable=False)  # 重试次数
    max_retries = Column(Integer, default=3, nullable=False)  # 最大重试次数
    retry_delay = Column(Integer, default=5, nullable=False)  # 重试延迟（秒）
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    progress = Column(Integer, default=0, nullable=False)  # 0-100
