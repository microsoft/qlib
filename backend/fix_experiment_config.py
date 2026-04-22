#!/usr/bin/env python3
"""
Fix the configuration of experiment 7
"""

from sqlalchemy import create_engine, text, update
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# Create database connection
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)

# Fix experiment 7 configuration
with engine.connect() as conn:
    # Get current configuration
    result = conn.execute(text('SELECT id, name, config FROM experiments WHERE id = 7'))
    row = result.fetchone()
    
    if row:
        print(f'Fixing experiment 7: {row[1]}')
        print(f'Current config type: {type(row[2])}')
        print(f'Current config: {row[2]}')
        
        try:
            # Fix 1: If config is a string, parse it to JSON
            if isinstance(row[2], str):
                config = json.loads(row[2])
            else:
                config = row[2]
            
            # Fix 2: Remove 'normalize' parameter from LinearModel
            if config.get('task', {}).get('model', {}).get('kwargs', {}).get('normalize') is not None:
                del config['task']['model']['kwargs']['normalize']
                print("Removed 'normalize' parameter from LinearModel")
            
            # Update experiment configuration using raw SQL
            conn.execute(
                text('UPDATE experiments SET config = :config WHERE id = 7'),
                {'config': json.dumps(config)}  # Convert config to JSON string for MySQL
            )
            conn.commit()
            
            print("\nFixed configuration:")
            print(json.dumps(config, indent=2))
            print("\nExperiment 7 configuration fixed successfully!")
            
        except Exception as e:
            print(f"\nError fixing configuration: {e}")
            import traceback
            traceback.print_exc()
    else:
        print('Experiment 7 not found')
