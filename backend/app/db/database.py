from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings

# Create SQLAlchemy engine with optimized connection pool settings
if settings.database_url.startswith("sqlite"):
    engine = create_engine(
        settings.database_url, 
        connect_args={"check_same_thread": False},
        pool_size=5,  # SQLite doesn't support real connection pooling, but we can set minimal settings
        max_overflow=0,
        pool_pre_ping=True,
        pool_recycle=3600
    )
else:
    # For PostgreSQL/MySQL, use optimized connection pool settings
    engine = create_engine(
        settings.database_url,
        pool_size=10,  # Number of connections to keep open in the pool
        max_overflow=20,  # Maximum number of connections to allow beyond pool_size
        pool_pre_ping=True,  # Check if connection is alive before using it
        pool_recycle=3600,  # Recycle connections after 1 hour
        pool_timeout=30,  # Timeout for getting a connection from the pool
        echo_pool=False,  # Disable pool logging for production
        execution_options={"isolation_level": "READ_COMMITTED"}  # Use appropriate isolation level
    )

# Create session local class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
