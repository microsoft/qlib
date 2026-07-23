#!/usr/bin/env python3
"""
Script to check database connection and tables
"""

from sqlalchemy.orm import Session
from app.db.database import engine, Base, get_db, settings
from app.models.experiment import Experiment
from app.models.log import ExperimentLog
from app.models.user import User

print(f"Database URL: {settings.database_url}")
print("=" * 50)

try:
    # Test database connection
    print("Testing database connection...")
    with engine.connect() as conn:
        print("✓ Database connection successful")
    
    # Create tables if they don't exist
    print("\nCreating tables if they don't exist...")
    Base.metadata.create_all(bind=engine)
    print("✓ Tables created/verified successfully")
    
    # Get database session
    db = next(get_db())
    
    # Check if tables exist
    print("\nChecking tables...")
    
    # Check users table
    users_count = db.query(User).count()
    print(f"✓ Users table: {users_count} users found")
    
    # Check experiments table
    experiments_count = db.query(Experiment).count()
    print(f"✓ Experiments table: {experiments_count} experiments found")
    
    # Check experiment_logs table
    logs_count = db.query(ExperimentLog).count()
    print(f"✓ Experiment_logs table: {logs_count} logs found")
    
    # Test querying experiments
    print("\nTesting experiment query...")
    experiments = db.query(Experiment).limit(5).all()
    print(f"✓ Queried {len(experiments)} experiments successfully")
    for exp in experiments:
        print(f"  - Experiment {exp.id}: {exp.name} (status: {exp.status})")
    
    # Test querying logs for an experiment
    if experiments:
        print(f"\nTesting logs query for experiment {experiments[0].id}...")
        logs = db.query(ExperimentLog).filter(ExperimentLog.experiment_id == experiments[0].id).limit(5).all()
        print(f"✓ Queried {len(logs)} logs successfully")
        for log in logs:
            print(f"  - Log: [{log.created_at.strftime('%Y-%m-%d %H:%M:%S')}] {log.message[:50]}...")
    
    print("\n" + "=" * 50)
    print("✓ All database checks passed!")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    try:
        db.close()
    except:
        pass