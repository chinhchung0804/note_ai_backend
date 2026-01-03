"""
Authentication wrapper for existing note processing endpoints
This module provides helper functions to add authentication and rate limiting
to the existing v1 API endpoints without breaking backward compatibility
"""

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
import os

from app.database.database import get_db
from app.database.models import User
from app.auth.security import get_current_active_user
from app.auth.rate_limiter import check_daily_note_limit, increment_note_count


async def get_optional_user(
    user_id: Optional[str] = None,
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Get user from user_id parameter (backward compatibility)
    This allows existing endpoints to work without authentication
    but still track users if user_id is provided
    """
    if not user_id:
        return None
    
    from app.services.db_service import db_service
    try:
        user = db_service.get_or_create_user(db, username=user_id)
        return user
    except Exception as e:
        print(f"Error getting user: {e}")
        return None


async def check_user_limits(
    user: Optional[User],
    db: Session,
    require_auth: bool = False
) -> None:
    """
    Check if user has reached their daily note limit
    
    Args:
        user: User object (can be None for backward compatibility)
        db: Database session
        require_auth: If True, authentication is required
    
    Raises:
        HTTPException: If user has reached their limit or auth is required but not provided
    """
    # If authentication is required but no user provided
    if require_auth and not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please login to use this feature."
        )
    
    # If no user, skip rate limiting (backward compatibility)
    if not user:
        return
    
    # Check rate limits
    try:
        check_daily_note_limit(db, user)
    except HTTPException:
        # Re-raise the exception from rate limiter
        raise


async def increment_user_note_count(
    user: Optional[User],
    db: Session
) -> None:
    """
    Increment user's note count after successful processing
    
    Args:
        user: User object (can be None for backward compatibility)
        db: Database session
    """
    if not user:
        return
    
    try:
        increment_note_count(db, user)
    except Exception as e:
        print(f"Error incrementing note count: {e}")


def get_ai_model_for_user(user: Optional[User]) -> str:
    """
    Get the appropriate AI model based on user's account type
    
    Args:
        user: User object (can be None for backward compatibility)
    
    Returns:
        Model name to use for AI processing
    """
    # Default model for non-authenticated users
    default_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    if not user:
        return default_model
    
    # Pro users get better models
    if user.account_type == "PRO":
        return "gpt-4"
    elif user.account_type == "ENTERPRISE":
        return "gpt-4"
    else:
        # Free users get the default model
        return default_model


def should_require_auth() -> bool:
    """
    Check if authentication should be required for all endpoints
    This can be controlled via environment variable
    
    Returns:
        True if authentication is required, False otherwise
    """
    return os.getenv("REQUIRE_AUTH", "false").lower() == "true"
