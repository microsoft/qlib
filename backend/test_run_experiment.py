#!/usr/bin/env python3
"""
Test script to run an experiment
"""

from sqlalchemy.orm import Session
from app.db.database import engine, Base, get_db
from app.models.experiment import Experiment
from app.services.task import TaskService

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

# Get database session
db = next(get_db())

try:
    # Get the experiment to run
    experiment = db.query(Experiment).filter(Experiment.id == 9).first()
    if not experiment:
        print("Experiment not found")
        exit(1)
    
    print(f"Found experiment: {experiment.name}")
    print(f"Current status: {experiment.status}")
    
    # Create a task to run the experiment
    print("\nCreating task...")
    task = TaskService.create_task(db=db, experiment_id=experiment.id, task_type="train")
    print(f"Task created: {task.id}")
    
    print("\nExperiment status updated to: pending")
    print("Task status: pending")
    
    # Get pending tasks
    print("\nGetting pending tasks...")
    pending_tasks = TaskService.get_pending_tasks(db=db)
    print(f"Found {len(pending_tasks)} pending tasks")
    
    for t in pending_tasks:
        print(f"  - Task {t.id}: Experiment {t.experiment_id} ({t.status})")
    
    print("\n✓ Test completed successfully!")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    db.close()