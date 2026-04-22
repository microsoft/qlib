#!/usr/bin/env python3
"""
Check the configuration of experiment 7
"""

from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# Create database connection
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)

# Check experiment 7 configuration
with engine.connect() as conn:
    result = conn.execute(text('SELECT id, name, config FROM experiments WHERE id = 7'))
    row = result.fetchone()
    
    if row:
        print(f'Experiment 7:')
        print(f'  ID: {row[0]}')
        print(f'  Name: {row[1]}')
        print(f'  Config:')
        print(json.dumps(row[2], indent=2))
    else:
        print('Experiment 7 not found')
