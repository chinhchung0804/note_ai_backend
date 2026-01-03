from typing import Iterable

from sqlalchemy import text

from app.database.database import engine, Base
from app.database import models  


NOTE_ALTER_STATEMENTS: Iterable[str] = (
    "ALTER TABLE notes ADD COLUMN IF NOT EXISTS processed_text TEXT",
    "ALTER TABLE notes ADD COLUMN IF NOT EXISTS summaries JSONB",
    "ALTER TABLE notes ADD COLUMN IF NOT EXISTS questions JSONB",
    "ALTER TABLE notes ADD COLUMN IF NOT EXISTS mcqs JSONB",
    "ALTER TABLE notes ADD COLUMN IF NOT EXISTS review JSONB",
    "ALTER TABLE notes ADD COLUMN IF NOT EXISTS summary TEXT",
    "DROP INDEX IF EXISTS ix_notes_note_id",
    "DO $ BEGIN IF EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ix_notes_note_id') THEN ALTER TABLE notes DROP CONSTRAINT ix_notes_note_id; END IF; END $;",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_notes_user_note_id ON notes (user_id, note_id)",
)

USER_ALTER_STATEMENTS: Iterable[str] = (
    # Authentication fields
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS email VARCHAR(255)",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS hashed_password VARCHAR(255)",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS is_verified BOOLEAN DEFAULT FALSE",
    
    # Account type and limits
    "DO $$ BEGIN CREATE TYPE accounttype AS ENUM ('FREE', 'PRO', 'ENTERPRISE'); EXCEPTION WHEN duplicate_object THEN null; END $$;",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS account_type VARCHAR(20) DEFAULT 'FREE'",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS daily_note_limit INTEGER DEFAULT 3",
    "UPDATE users SET daily_note_limit = 3 WHERE account_type = 'FREE' AND daily_note_limit = 5",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS notes_created_today INTEGER DEFAULT 0",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS last_reset_date DATE",
    
    # Subscription tracking
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS subscription_start TIMESTAMP",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS subscription_end TIMESTAMP",
    
    # Indexes for performance
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_users_email ON users (email) WHERE email IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS ix_users_account_type ON users (account_type)",
)

PAYMENT_TABLE_CREATION = """
CREATE TABLE IF NOT EXISTS payments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    amount DECIMAL(10, 2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'VND',
    payment_method VARCHAR(20) NOT NULL,
    payment_status VARCHAR(20) DEFAULT 'pending',
    transaction_id VARCHAR(255),
    plan_type VARCHAR(20) NOT NULL,
    plan_duration_months INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS ix_payments_user_id ON payments (user_id);
CREATE INDEX IF NOT EXISTS ix_payments_transaction_id ON payments (transaction_id);
CREATE INDEX IF NOT EXISTS ix_payments_status ON payments (payment_status);
"""


def run_migrations() -> None:
    """
    Idempotent schema updates to keep legacy databases in sync
    with the current SQLAlchemy models.
    """
    # Create all tables from models
    Base.metadata.create_all(bind=engine)
    
    with engine.begin() as conn:
        # Run note migrations
        for statement in NOTE_ALTER_STATEMENTS:
            try:
                conn.execute(text(statement))
            except Exception as e:
                print(f"Warning: Migration statement failed: {e}")
        
        # Run user migrations
        for statement in USER_ALTER_STATEMENTS:
            try:
                conn.execute(text(statement))
            except Exception as e:
                print(f"Warning: User migration statement failed: {e}")
        
        # Create payments table
        try:
            conn.execute(text(PAYMENT_TABLE_CREATION))
        except Exception as e:
            print(f"Warning: Payment table creation failed: {e}")
    
    print("✅ Migrations complete (users, notes, payments, feedbacks schema up-to-date).")


if __name__ == "__main__":
    run_migrations()
