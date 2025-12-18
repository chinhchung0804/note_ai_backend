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
    # Chuẩn hóa unique: note_id chỉ unique trong phạm vi user_id
    # 1) Bỏ unique/index cũ trên note_id nếu tồn tại
    "DROP INDEX IF EXISTS ix_notes_note_id",
    "DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ix_notes_note_id') THEN ALTER TABLE notes DROP CONSTRAINT ix_notes_note_id; END IF; END $$;",
    # 2) Tạo unique index mới theo (user_id, note_id)
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_notes_user_note_id ON notes (user_id, note_id)",
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

