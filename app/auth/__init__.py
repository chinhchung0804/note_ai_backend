"""
Authentication module for NotallyX backend
Handles user registration, login, JWT tokens, and password management
"""

from .security import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user,
    get_current_active_user
)
from .schemas import (
    UserRegister,
    UserLogin,
    UserResponse,
    UserUpdate,
    Token,
    PasswordChange,
    AccountLimits
)
from .rate_limiter import check_daily_note_limit, increment_note_count

__all__ = [
    "get_password_hash",
    "verify_password",
    "create_access_token",
    "get_current_user",
    "get_current_active_user",
    "UserRegister",
    "UserLogin",
    "UserResponse",
    "UserUpdate",
    "Token",
    "PasswordChange",
    "AccountLimits",
    "check_daily_note_limit",
    "increment_note_count"
]
