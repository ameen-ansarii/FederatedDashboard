from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from app.core.config import settings


class Base(DeclarativeBase):
    pass


if settings.ENVIRONMENT == "testing" or "sqlite" in settings.DATABASE_URL:
    engine = create_engine(
        "sqlite:///:memory:" if settings.ENVIRONMENT == "testing" else settings.DATABASE_URL,
        connect_args={"check_same_thread": False},
    )
else:
    try:
        engine = create_engine(
            settings.DATABASE_URL,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
    except Exception as e:
        # Fallback to local SQLite when PostgreSQL driver or host is unavailable
        print(f"[Database] Notice: falling back to SQLite ({e})")
        engine = create_engine("sqlite:///./storage_data/local_dev.db", connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)




def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
