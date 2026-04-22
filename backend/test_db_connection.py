#!/usr/bin/env python3
"""
Test script to verify database connection
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables from .env file
load_dotenv()

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("Error: DATABASE_URL environment variable not set")
    exit(1)

print(f"Testing database connection with URL: {DATABASE_URL}")

try:
    # Create engine
    engine = create_engine(DATABASE_URL)
    
    # Test connection
    with engine.connect() as conn:
        print("✓ Successfully connected to the database!")
        
        # Get database name
        result = conn.execute(text("SELECT DATABASE()"))
        db_name = result.fetchone()[0]
        print(f"✓ Connected to database: {db_name}")
        
        # Get server version
        result = conn.execute(text("SELECT VERSION()"))
        version = result.fetchone()[0]
        print(f"✓ MySQL server version: {version}")
        
except Exception as e:
    print(f"✗ Failed to connect to database: {e}")
    exit(1)

print("\nAll tests passed! Database connection is working correctly.")
