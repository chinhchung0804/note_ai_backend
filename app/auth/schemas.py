"""
Pydantic schemas for authentication
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, validator


class UserRegister(BaseModel):
    """Schema for user registration"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    
    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username must be alphanumeric (can include _ and -)')
        return v
    
    @validator('password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        return v


class UserLogin(BaseModel):
    """Schema for user login"""
    username: str
    password: str


class Token(BaseModel):
    """Schema for JWT token response"""
    access_token: str
    token_type: str = "bearer"
    user: 'UserResponse'


class UserResponse(BaseModel):
    """Schema for user response"""
    id: str
    username: str
    email: str
    account_type: str
    is_active: bool
    is_verified: bool
    daily_note_limit: int
    notes_created_today: int
    subscription_end: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    """Schema for updating user profile"""
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=8, max_length=100)


class PasswordChange(BaseModel):
    """Schema for changing password"""
    old_password: str
    new_password: str = Field(..., min_length=8, max_length=100)
    
    @validator('new_password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        return v


class AccountLimits(BaseModel):
    """Schema for account limits response"""
    account_type: str
    daily_note_limit: int
    notes_created_today: int
    remaining_notes: int
    limit_resets_at: datetime
    
    class Config:
        from_attributes = True
