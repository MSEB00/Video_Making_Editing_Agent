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


def _resolve_database_url() -> str:
    """Resolve DB URL – fallback to local SQLite file under data/."""
    return os.getenv("DATABASE_URL") or f"sqlite:///{Path(__file__).parents[2] / 'data' / 'agent.db'}"


DATABASE_URL = _resolve_database_url()
engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))


def reset_engine() -> None:
    """Rebind engine and sessions to the CURRENT DATABASE_URL env value.

    The module-level engine is created at import time; tests and long-lived
    processes that change DATABASE_URL afterwards must call this. SessionLocal
    is reconfigured in place, so every already-imported reference (e.g. inside
    the orchestrator) transparently follows the new binding. The previous
    engine is disposed so its pooled connections release their file handles —
    on Windows an open SQLite handle blocks file deletion.
    """
    global DATABASE_URL, engine
    SessionLocal.remove()
    try:
        engine.dispose()
    except Exception:
        pass
    DATABASE_URL = _resolve_database_url()
    engine = create_engine(DATABASE_URL, echo=False, future=True)
    SessionLocal.configure(bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)
