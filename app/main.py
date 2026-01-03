from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import signal
import sys
import asyncio
load_dotenv()

from app.api.v1.router import router as api_router
from app.api.routes.auth import router as auth_router
from app.api.routes.payment import router as payment_router

app = FastAPI(
    title="NotallyX AI Backend",
    version="2.0.0",
    description="AI-powered note processing with authentication and subscription management"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8080",
        "https://notallyx.app",
        "*"  # For development - restrict in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_router, prefix="/api/v1", tags=["Notes Processing"])
app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(payment_router, prefix="/api/payment", tags=["Payment & Subscription"])

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
    try:
        pass
    except Exception as e:
        print(f"⚠️  Warning during shutdown: {e}")
    print("✅ Server shutdown complete")

@app.get('/')
def root():
    return {'message':'Note Summarizer AI Backend running', 'llm':'Google Gemini via LangChain'}
