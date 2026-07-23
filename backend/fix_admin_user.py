#!/usr/bin/env python3
"""
Script to fix admin user in the database
"""

from sqlalchemy.orm import Session
from app.db.database import engine, Base, get_db
from app.models.user import User
from app.services.auth import get_password_hash

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

# Get database session
db = next(get_db())

try:
    # Check if admin user exists
    admin_user = db.query(User).filter(User.username == "admin").first()
    
    if admin_user:
        print(f"Admin user found: {admin_user.username}")
        print(f"Current role: {admin_user.role}")
        print(f"Disabled: {admin_user.disabled}")
        
        # Update admin user with correct password and role
        admin_user.password_hash = get_password_hash("admin123")
        admin_user.role = "admin"
        admin_user.disabled = False
        db.commit()
        print("Admin user updated successfully")
    else:
        # Create new admin user
        new_admin = User(
            username="admin",
            password_hash=get_password_hash("admin123"),
            role="admin",
            disabled=False
        )
        db.add(new_admin)
        db.commit()
        print("Admin user created successfully")
        
    # Verify the fix
    test_user = db.query(User).filter(User.username == "admin").first()
    print(f"\nVerification:")
    print(f"Username: {test_user.username}")
    print(f"Role: {test_user.role}")
    print(f"Disabled: {test_user.disabled}")
    print("Admin user fix completed successfully")
    
except Exception as e:
    print(f"Error: {e}")
    db.rollback()
finally:
    db.close()