"""
Rate limiter for FREE vs PRO accounts
Also handles AI model selection based on account type
"""
from datetime import datetime, date
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from enum import Enum

from app.database.models import User, AccountType


class AIModel(str, Enum):
    """AI Models available for different account types"""
    GEMINI_FLASH = "gemini-2.5-flash"
    GPT4O_MINI = "gpt-4o-mini"
    GPT4 = "gpt-4"


def get_model_for_account(account_type: AccountType) -> str:
    """
    Get AI model based on account type
    
    FREE: Gemini 2.5 Flash (free, 20 RPD limit → 3 notes/day with buffer)
    PRO: GPT-4o-mini (paid, unlimited, good quality, ~$0.003/note)
    ENTERPRISE: GPT-4 (paid, unlimited, best quality, ~$0.02/note)
    """
    model_mapping = {
        AccountType.FREE: AIModel.GEMINI_FLASH.value,
        AccountType.PRO: AIModel.GPT4O_MINI.value,
        AccountType.ENTERPRISE: AIModel.GPT4.value
    }
    return model_mapping.get(account_type, AIModel.GEMINI_FLASH.value)


def get_daily_limit_for_account(account_type: AccountType) -> int:
    """
    Get daily note limit based on account type
    
    FREE: 3 notes/day (safe buffer for Gemini 20 RPD limit)
          Calculation: 20 RPD ÷ 4 requests/note = 5 notes max
          Using 3 for 40% buffer (errors, retries, multiple users)
    PRO: Unlimited (-1)
    ENTERPRISE: Unlimited (-1)
    """
    limits = {
        AccountType.FREE: 3,        # Conservative limit with buffer
        AccountType.PRO: -1,        # Unlimited
        AccountType.ENTERPRISE: -1  # Unlimited
    }
    return limits.get(account_type, 3)


def check_daily_note_limit(db: Session, user: User) -> None:
    """
    Check if user has exceeded their daily note limit
    
    Raises HTTPException if limit exceeded
    """
    # PRO and ENTERPRISE users have unlimited notes
    if user.account_type in [AccountType.PRO, AccountType.ENTERPRISE]:
        return
    
    # Check if we need to reset daily counter
    today = datetime.utcnow().date()
    last_reset = user.last_reset_date.date() if user.last_reset_date else None
    
    if last_reset != today:
        # Reset counter for new day
        user.notes_created_today = 0
        user.last_reset_date = datetime.utcnow()
        db.commit()
    
    # Check limit
    if user.notes_created_today >= user.daily_note_limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Bạn đã đạt giới hạn {user.daily_note_limit} ghi chú/ngày. "
                   f"Nâng cấp lên PRO để không giới hạn! "
                   f"(Giới hạn sẽ reset vào {(datetime.utcnow().date() + timedelta(days=1)).isoformat()})"
        )


def increment_note_count(db: Session, user: User) -> None:
    """
    Increment user's daily note count after successful note creation
    """
    # Only increment for FREE users (PRO/ENTERPRISE still tracked but not limited)
    user.notes_created_today += 1
    db.commit()


def reset_daily_limits(db: Session) -> int:
    """
    Reset daily limits for all users
    Should be called by a cron job at midnight UTC
    
    Returns number of users reset
    """
    from app.database.models import User
    
    today = datetime.utcnow().date()
    
    # Find users who need reset
    users_to_reset = db.query(User).filter(
        (User.last_reset_date == None) | 
        (User.last_reset_date < today)
    ).all()
    
    count = 0
    for user in users_to_reset:
        user.notes_created_today = 0
        user.last_reset_date = datetime.utcnow()
        count += 1
    
    db.commit()
    return count


def get_remaining_notes(user: User) -> int:
    """
    Get remaining notes for today
    
    Returns:
        -1 for unlimited (PRO/ENTERPRISE)
        Number of remaining notes for FREE
    """
    if user.account_type in [AccountType.PRO, AccountType.ENTERPRISE]:
        return -1
    
    # Check if need reset
    today = datetime.utcnow().date()
    last_reset = user.last_reset_date.date() if user.last_reset_date else None
    
    if last_reset != today:
        return user.daily_note_limit
    
    remaining = user.daily_note_limit - user.notes_created_today
    return max(0, remaining)


def get_limit_reset_time() -> datetime:
    """
    Get the time when limits will reset (midnight UTC)
    """
    from datetime import timedelta
    tomorrow = datetime.utcnow().date() + timedelta(days=1)
    return datetime.combine(tomorrow, datetime.min.time())
