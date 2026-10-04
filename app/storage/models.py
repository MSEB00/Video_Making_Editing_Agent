"""
app/storage/models.py
----------------------
SQLAlchemy ORM definitions for core entities.
"""
from __future__ import annotations

import datetime as dt
from sqlalchemy import Column, DateTime, Integer, String, Text, Float
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Job(Base):
    """High‑level video processing job."""
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    status = Column(String(20), default="queued", nullable=False)
    input_path = Column(Text, nullable=False)
    output_path = Column(Text, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)
    extra_metadata = Column(Text, nullable=True)  # renamed to avoid conflict

class Media(Base):
    """Tracks media assets (raw clips, thumbnails, etc.)."""
    __tablename__ = "media"
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, nullable=False)
    kind = Column(String(30), nullable=False)
    path = Column(Text, nullable=False)
    duration = Column(Float, nullable=True)
    hash_sha256 = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

class Event(Base):
    """Detected game events (kill, ace, clutch…)."""
    __tablename__ = "events"
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, nullable=False)
    type = Column(String(30), nullable=False)
    timestamp = Column(Float, nullable=False)
    confidence = Column(Float, nullable=True)
    payload = Column(Text, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
