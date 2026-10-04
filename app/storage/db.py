"""
app/storage/db.py
-----------------
SQLAlchemy engine and session factory.
"""
from __future__ import annotations
import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from app.storage.models import Base

# Resolve DB URL – fallback to local SQLite file
DATABASE_URL = os.getenv("DATABASE_URL") or f"sqlite:///" + str(Path(__file__).parents[2] / "data" / "agent.db")

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
