from app.database.database import init_db, engine
from app.database.models import User, Note

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Database initialized successfully!")
    print("Tables created: users, notes")

