#!/usr/bin/env python3
"""
Reset the admin user password
"""

from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
from app.services.auth import get_password_hash

# Load environment variables
load_dotenv()

# Create database connection
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)

# Reset admin password
with engine.connect() as conn:
    # Generate new password hash
    new_password = "admin123"
    hashed_password = get_password_hash(new_password)
    
    # Update admin user password
    conn.execute(
        text('UPDATE users SET password_hash = :password_hash WHERE username = "admin"'),
        {'password_hash': hashed_password}
    )
    conn.commit()
    
    print(f"Admin password has been reset to: {new_password}")
    print(f"New password hash: {hashed_password}")
    
    # Verify the update
    result = conn.execute(text('SELECT username, password_hash FROM users WHERE username = "admin"'))
    user = result.fetchone()
    if user:
        print(f"Admin user password updated successfully")
