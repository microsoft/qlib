#!/usr/bin/env python3
"""
Fix the user table by adding missing columns.
"""

import pymysql

# Database connection settings
DB_CONFIG = {
    'host': 'rm-bp146d0y4vo46bn72co.mysql.rds.aliyuncs.com',
    'user': 'hoo',
    'password': 'Moshou99',
    'database': 'qlib_ai',
    'port': 3306
}

def fix_user_table():
    """Fix the user table by adding missing columns."""
    try:
        # Connect to database
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print("Connected to database successfully")
        
        # Check current table structure
        cursor.execute("DESCRIBE users")
        columns = cursor.fetchall()
        column_names = [col[0] for col in columns]
        print(f"Current columns: {column_names}")
        
        # Add missing columns
        missing_columns = []
        
        # Check if email_verified column exists
        if 'email_verified' not in column_names:
            cursor.execute("ALTER TABLE users ADD COLUMN email_verified BOOLEAN DEFAULT FALSE")
            missing_columns.append('email_verified')
        
        # Check if verification_token column exists
        if 'verification_token' not in column_names:
            cursor.execute("ALTER TABLE users ADD COLUMN verification_token VARCHAR(255)")
            missing_columns.append('verification_token')
        
        # Check if verification_token_expiry column exists
        if 'verification_token_expiry' not in column_names:
            cursor.execute("ALTER TABLE users ADD COLUMN verification_token_expiry DATETIME")
            missing_columns.append('verification_token_expiry')
        
        # Check if last_login column exists
        if 'last_login' not in column_names:
            cursor.execute("ALTER TABLE users ADD COLUMN last_login DATETIME")
            missing_columns.append('last_login')
        
        # Check if updated_at column exists
        if 'updated_at' not in column_names:
            cursor.execute("ALTER TABLE users ADD COLUMN updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP")
            missing_columns.append('updated_at')
        
        # Check if password_reset_token column exists
        if 'password_reset_token' not in column_names:
            cursor.execute("ALTER TABLE users ADD COLUMN password_reset_token VARCHAR(255)")
            missing_columns.append('password_reset_token')
        
        # Check if password_reset_expiry column exists
        if 'password_reset_expiry' not in column_names:
            cursor.execute("ALTER TABLE users ADD COLUMN password_reset_expiry DATETIME")
            missing_columns.append('password_reset_expiry')
        
        # Commit changes
        conn.commit()
        print(f"Added missing columns: {missing_columns}")
        
        # Verify the changes
        cursor.execute("DESCRIBE users")
        columns = cursor.fetchall()
        column_names = [col[0] for col in columns]
        print(f"Updated columns: {column_names}")
        
        # Check if admin user exists
        cursor.execute("SELECT * FROM users WHERE username = 'admin'")
        admin_user = cursor.fetchone()
        if not admin_user:
            # Create admin user
            cursor.execute("INSERT INTO users (username, password_hash, role, disabled, email_verified) VALUES ('admin', '$pbkdf2-sha256$29000$9QJ6iLJz5rRvZ5rRvZ5rRv$Z5rRvZ5rRvZ5rRvZ5rRvZ5rRvZ5rRvZ5rRvZ5rRv', 'admin', 0, 1)")
            conn.commit()
            print("Created admin user")
        
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    fix_user_table()
