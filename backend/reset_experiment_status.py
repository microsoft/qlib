#!/usr/bin/env python3
"""
Reset the status of experiment 7 to pending
"""

from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create database connection
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)

# Reset experiment 7 status
with engine.connect() as conn:
    # Reset experiment status to pending
    conn.execute(
        text('UPDATE experiments SET status = "pending", progress = 0.0 WHERE id = 7')
    )
    conn.commit()
    
    # Create a new task for the experiment
    conn.execute(
        text('INSERT INTO tasks (experiment_id, task_type, status, priority, progress, created_at) VALUES (7, "train", "pending", 1, 0.0, NOW())')
    )
    conn.commit()
    
    print("Experiment 7 status reset to 'pending' and a new task has been created")
    
    # Verify the changes
    result = conn.execute(text('SELECT id, name, status, progress FROM experiments WHERE id = 7'))
    row = result.fetchone()
    if row:
        print(f'Updated experiment 7: id={row[0]}, name={row[1]}, status={row[2]}, progress={row[3]}%')
