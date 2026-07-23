#!/usr/bin/env python3
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.user import User

# Create database engine
engine = create_engine('mysql+pymysql://hoo:Moshou99@rm-bp146d0y4vo46bn72co.mysql.rds.aliyuncs.com/qlib_ai')
Session = sessionmaker(bind=engine)
db = Session()

try:
    # Query all users
    users = db.query(User).all()
    print(f"Found {len(users)} users:")
    for user in users:
        print(f"  ID: {user.id}")
        print(f"  Username: {user.username}")
        print(f"  Role: {user.role}")
        print(f"  Disabled: {user.disabled}")
        print(f"  Email Verified: {user.email_verified}")
        print(f"  Password Hash: {user.password_hash[:20]}...")
        print("  --")
finally:
    db.close()
