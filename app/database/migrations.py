"""
Simple ad-hoc migrations for PostgreSQL.

Run: `python -m app.database.migrations`
"""
from typing import Iterable

from sqlalchemy import text

from app.database.database import engine, Base
from app.database import models  # noqa: F401  # ensure metadata is loaded


NOTE_ALTER_STATEMENTS: Iterable[str] = (
    "ALTER TABLE notes ADD COLUMN IF NOT EXISTS processed_text TEXT",
    "ALTER TABLE notes ADD COLUMN IF NOT EXISTS summaries JSONB",
    "ALTER TABLE notes ADD COLUMN IF NOT EXISTS questions JSONB",
    "ALTER TABLE notes ADD COLUMN IF NOT EXISTS mcqs JSONB",
    "ALTER TABLE notes ADD COLUMN IF NOT EXISTS review JSONB",
    "ALTER TABLE notes ADD COLUMN IF NOT EXISTS summary TEXT",
)


def run_migrations() -> None:
    """
    Idempotent schema updates to keep legacy databases in sync
    with the current SQLAlchemy models.
    """
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        for statement in NOTE_ALTER_STATEMENTS:
            conn.execute(text(statement))
    print("✅ Migrations complete (notes + feedbacks schema up-to-date).")


if __name__ == "__main__":
    run_migrations()

