"""
Authentication routes
"""
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import User
from app.auth.schemas import UserRegister, UserLogin, Token, UserResponse, UserUpdate, PasswordChange
from app.auth.security import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_active_user
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """
    Register new user
    
    - **username**: Unique username (3-50 characters, alphanumeric)
    - **email**: Valid email address
    - **password**: Strong password (min 8 chars, 1 digit, 1 uppercase)
    """
    # Check if username exists
    existing_user = db.query(User).filter(User.username == user_data.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email exists
    existing_email = db.query(User).filter(User.email == user_data.email).first()
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password),
        is_active=True,
        is_verified=False,  # Can implement email verification later
        account_type="free",
        daily_note_limit=5,
        notes_created_today=0,
        last_reset_date=datetime.utcnow()
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Create access token
    access_token = create_access_token(data={"sub": str(new_user.id)})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": new_user.to_dict()
    }


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Login with username and password
    
    Returns JWT access token
    """
    # Find user
    user = db.query(User).filter(User.username == form_data.username).first()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    # Create access token
    access_token = create_access_token(data={"sub": str(user.id)})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user.to_dict()
    }


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user information"""
    return current_user.to_dict()


@router.put("/me", response_model=UserResponse)
async def update_user_profile(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update user profile"""
    if user_update.email:
        # Check if email already exists
        existing_email = db.query(User).filter(
            User.email == user_update.email,
            User.id != current_user.id
        ).first()
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already in use"
            )
        current_user.email = user_update.email
    
    if user_update.password:
        current_user.hashed_password = get_password_hash(user_update.password)
    
    db.commit()
    db.refresh(current_user)
    
    return current_user.to_dict()


@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Change user password"""
    # Verify old password
    if not verify_password(password_data.old_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect old password"
        )
    
    # Update password
    current_user.hashed_password = get_password_hash(password_data.new_password)
    db.commit()
    
    return {"message": "Password changed successfully"}


@router.get("/account-limits")
async def get_account_limits(
    current_user: User = Depends(get_current_active_user)
):
    """Get current account limits and usage"""
    from app.auth.rate_limiter import get_account_limits
    
    limits = get_account_limits(current_user.account_type)
    
    return {
        "account_type": current_user.account_type.value,
        "limits": limits,
        "usage": {
            "notes_created_today": current_user.notes_created_today,
            "daily_limit": current_user.daily_note_limit
        },
        "subscription": {
            "is_active": current_user.account_type == "pro",
            "end_date": current_user.subscription_end.isoformat() if current_user.subscription_end else None
        }
    }


@router.get("/account-benefits")
async def get_account_benefits():
    """
    Get detailed benefits for all account types.
    
    Returns information about FREE, PRO, and ENTERPRISE plans including:
    - Features available
    - Daily note limits
    - AI models used
    - Vocab features
    - Pricing
    """
    from app.core.feature_config import get_account_benefits
    
    return get_account_benefits()


@router.get("/my-benefits")
async def get_my_benefits(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get benefits for current user's account type.
    
    Returns detailed information about what features are available
    for the current user based on their account type.
    """
    from app.core.feature_config import get_account_benefits, get_enabled_vocab_features
    
    all_benefits = get_account_benefits()
    account_type = current_user.account_type.value
    
    my_benefits = all_benefits.get(account_type, all_benefits["free"])
    my_benefits["enabled_vocab_features"] = get_enabled_vocab_features(account_type)
    
    return {
        "account_type": account_type,
        "benefits": my_benefits,
        "usage": {
            "notes_created_today": current_user.notes_created_today,
            "daily_limit": current_user.daily_note_limit,
            "remaining_notes": current_user.daily_note_limit - current_user.notes_created_today if current_user.daily_note_limit > 0 else -1
        }
    }
