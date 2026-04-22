#!/usr/bin/env python3
"""
Update the admin user password to ensure it can login correctly.
"""

import pymysql
from app.services.auth import get_password_hash

# Database connection settings
DB_CONFIG = {
    'host': 'rm-bp146d0y4vo46bn72co.mysql.rds.aliyuncs.com',
    'user': 'hoo',
    'password': 'Moshou99',
    'database': 'qlib_ai',
    'port': 3306
}

def update_admin_password():
    """Update the admin user password."""
    try:
        # Connect to database
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print("Connected to database successfully")
        
        # Generate password hash for admin123
        admin_password = "admin123"
        password_hash = get_password_hash(admin_password)
        print(f"Generated password hash for '{admin_password}': {password_hash}")
        
        # Check if admin user exists
        cursor.execute("SELECT * FROM users WHERE username = 'admin'")
        admin_user = cursor.fetchone()
        
        if admin_user:
            # Update existing admin user
            cursor.execute(
                "UPDATE users SET password_hash = %s, role = 'admin', disabled = 0, email_verified = 1 WHERE username = 'admin'",
                (password_hash,)
            )
            conn.commit()
            print("Updated admin user password")
        else:
            # Create new admin user
            cursor.execute(
                "INSERT INTO users (username, password_hash, role, disabled, email_verified) VALUES (%s, %s, 'admin', 0, 1)",
                ('admin', password_hash)
            )
            conn.commit()
            print("Created admin user")
        
        # Verify the changes
        cursor.execute("SELECT username, role, disabled, email_verified FROM users WHERE username = 'admin'")
        admin_user = cursor.fetchone()
        if admin_user:
            print(f"Admin user status: username={admin_user[0]}, role={admin_user[1]}, disabled={admin_user[2]}, email_verified={admin_user[3]}")
        
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    update_admin_password()
