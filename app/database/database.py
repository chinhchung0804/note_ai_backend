"""
Database Configuration - PostgreSQL với SQLAlchemy
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Database URL from environment variables
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://user:password@db:5432/note_ai'
)

# create engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  
    pool_size=10,
    max_overflow=20
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class cho models
Base = declarative_base()

def get_db():
    """
    Dependency để lấy database session
    Sử dụng trong FastAPI endpoints
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """
    Khởi tạo database - tạo tables
    """
    from app.database.models import User, Note, Feedback
    Base.metadata.create_all(bind=engine)

