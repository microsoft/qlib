from app.db.database import engine, Base, SessionLocal
from app.models.user import User
from app.models.experiment import Experiment
from app.models.model_version import ModelVersion
from app.models.config import Config
from app.models.factor import Factor, FactorGroup

# Drop all existing tables
Base.metadata.drop_all(bind=engine)

# Create all tables
Base.metadata.create_all(bind=engine)

print("Database tables recreated successfully!")