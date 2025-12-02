from fastapi import FastAPI
from dotenv import load_dotenv
import os
import signal
import sys
import asyncio
load_dotenv()

from app.api.v1.router import router as api_router

app = FastAPI(title="Note Summarizer AI Backend", version="1.0.0")
app.include_router(api_router, prefix="/api/v1")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """
    Khởi tạo database khi app start
    Tạo tables nếu chưa có
    """
    try:
        from app.database.database import init_db
        init_db()
        print("✅ Database initialized successfully!")
    except Exception as e:
        print(f"⚠️  Warning: Database initialization failed: {e}")
        print("Make sure PostgreSQL is running and DATABASE_URL is correct")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup khi app shutdown
    """
    print("🛑 Shutting down server...")
    # Đóng database connections nếu cần
    try:
        # Có thể thêm cleanup logic ở đây nếu cần
        pass
    except Exception as e:
        print(f"⚠️  Warning during shutdown: {e}")
    print("✅ Server shutdown complete")

@app.get('/')
def root():
    return {'message':'Note Summarizer AI Backend running', 'llm':'Google Gemini via LangChain'}
