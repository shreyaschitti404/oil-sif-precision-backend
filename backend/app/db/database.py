from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import settings


engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


def test_database_connection():
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    return True