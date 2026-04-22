#!/usr/bin/env python3
"""
Log model for optimized log storage and querying
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.db.database import Base

class ExperimentLog(Base):
    """实验日志模型"""
    __tablename__ = "experiment_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False, index=True)
    message = Column(Text, nullable=False)
    level = Column(String(20), default="info", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    def __repr__(self):
        return f"<ExperimentLog(experiment_id={self.experiment_id}, level={self.level}, message={self.message[:50]}...)>"
