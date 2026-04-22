#!/usr/bin/env python3
"""
Create experiment_logs table in the database
"""

from app.db.database import engine, Base
from app.models.experiment import Experiment
from app.models.log import ExperimentLog
from app.models.user import User
from app.models.task import Task

# Create all tables in the correct order
print("Creating all tables...")
Base.metadata.create_all(bind=engine)
print("All tables created successfully!")
