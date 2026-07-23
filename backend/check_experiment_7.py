#!/usr/bin/env python3
"""
Check the status of experiment 7
"""

from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create database connection
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)

# Check experiment 7 status
with engine.connect() as conn:
    result = conn.execute(text('SELECT id, name, status, progress, logs FROM experiments WHERE id = 7'))
    row = result.fetchone()
    
    if row:
        print(f'Experiment 7:')
        print(f'  ID: {row[0]}')
        print(f'  Name: {row[1]}')
        print(f'  Status: {row[2]}')
        print(f'  Progress: {row[3]}%')
        print(f'  Logs:')
        print(row[4] if row[4] else '  No logs')
    else:
        print('Experiment 7 not found')
