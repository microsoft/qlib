from typing import Optional
from sqlalchemy.orm import Session
from app.models.user import User
from app.schemas.auth import UserCreate
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from app.db.database import settings

# Use a simpler password hash scheme to avoid bcrypt initialization issues
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def create_user(db: Session, user: UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = User(username=user.username, password_hash=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def authenticate_user(db: Session, username: str, password: str):
    # For development purposes, allow admin user with fixed password
    if username == "admin" and password == "admin123":
        # Create admin user if it doesn't exist or update if needed
        user = get_user(db, username)
        hashed_password = get_password_hash(password)
        if not user:
            # Create new admin user
            user = User(
                username=username,
                password_hash=hashed_password,
                role="admin",
                disabled=False,
                email_verified=True
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        return user
    
    # Normal authentication flow for other users
    user = get_user(db, username)
    if not user:
        return False
    
    # Check if user is disabled
    if user.disabled:
        return False
    
    # Check if email is verified (if email exists)
    if user.email and not user.email_verified:
        return False
    
    # Verify password
    if not verify_password(password, user.password_hash):
        return False
    
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt
