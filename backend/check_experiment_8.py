#!/usr/bin/env python3
"""
Check the status and logs of experiment 8
"""

from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create database connection
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)

# Check experiment 8 status
with engine.connect() as conn:
    # Get experiment status
    result = conn.execute(text('SELECT id, name, status, progress FROM experiments WHERE id = 8'))
    experiment = result.fetchone()
    
    if experiment:
        print(f'Experiment 8:')
        print(f'  ID: {experiment[0]}')
        print(f'  Name: {experiment[1]}')
        print(f'  Status: {experiment[2]}')
        print(f'  Progress: {experiment[3]}%')
        
        # Get logs count
        result = conn.execute(text('SELECT COUNT(*) FROM experiment_logs WHERE experiment_id = 8'))
        log_count = result.fetchone()[0]
        print(f'  Logs count: {log_count}')
        
        # Get last 10 logs
        print(f'  Last 10 logs:')
        result = conn.execute(
            text('SELECT created_at, message FROM experiment_logs WHERE experiment_id = 8 ORDER BY created_at DESC LIMIT 10')
        )
        logs = result.fetchall()
        for log in reversed(logs):
            timestamp = log[0].strftime('%Y-%m-%d %H:%M:%S')
            print(f'    [{timestamp}] {log[1]}')
    else:
        print('Experiment 8 not found')
