from datetime import datetime
import uuid
from enum import Enum

from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer, JSON, UniqueConstraint, Boolean, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship

from app.database.database import Base


class AccountType(str, Enum):
    """Loại tài khoản"""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class User(Base):
    """
    User Model - Lưu thông tin người dùng
    """
    __tablename__ = 'users'
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    
    # Authentication
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Account Type & Limits
    account_type = Column(SQLEnum(AccountType), default=AccountType.FREE, nullable=False)
    daily_note_limit = Column(Integer, default=3, nullable=False)  # Free: 3 notes/day (Gemini limit), Pro: unlimited (-1)
    notes_created_today = Column(Integer, default=0, nullable=False)
    last_reset_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Subscription (for Pro users)
    subscription_start = Column(DateTime, nullable=True)
    subscription_end = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    notes = relationship("Note", back_populates="user", cascade="all, delete-orphan")
    feedbacks = relationship("Feedback", back_populates="user", cascade="all, delete-orphan")
    payments = relationship("Payment", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, account_type={self.account_type})>"
    
    def to_dict(self):
        """Convert model thành dict để trả về API"""
        return {
            'id': str(self.id),
            'username': self.username,
            'email': self.email,
            'account_type': self.account_type.value,
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'daily_note_limit': self.daily_note_limit,
            'notes_created_today': self.notes_created_today,
            'subscription_end': self.subscription_end.isoformat() if self.subscription_end else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class Payment(Base):
    """
    Payment Model - Lưu lịch sử thanh toán
    """
    __tablename__ = 'payments'
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PGUUID(as_uuid=True), ForeignKey('users.id'), nullable=False, index=True)
    
    # Payment info
    amount = Column(Integer, nullable=False)  # Số tiền (VND hoặc USD cents)
    currency = Column(String(10), default='VND', nullable=False)
    payment_method = Column(String(50), nullable=False)  # vnpay, stripe, momo
    
    # Transaction details
    transaction_id = Column(String(255), unique=True, nullable=False, index=True)
    status = Column(String(50), default='pending', nullable=False)  # pending, completed, failed, refunded
    
    # Subscription details
    subscription_months = Column(Integer, nullable=False)  # Số tháng đăng ký
    subscription_start = Column(DateTime, nullable=True)
    subscription_end = Column(DateTime, nullable=True)
    
    # Metadata
    payment_data = Column(JSON, nullable=True)  # Lưu response từ payment gateway
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    user = relationship("User", back_populates="payments")
    
    def __repr__(self):
        return f"<Payment(id={self.id}, user_id={self.user_id}, amount={self.amount}, status={self.status})>"
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'user_id': str(self.user_id),
            'amount': self.amount,
            'currency': self.currency,
            'payment_method': self.payment_method,
            'transaction_id': self.transaction_id,
            'status': self.status,
            'subscription_months': self.subscription_months,
            'subscription_start': self.subscription_start.isoformat() if self.subscription_start else None,
            'subscription_end': self.subscription_end.isoformat() if self.subscription_end else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class Note(Base):
    """
    Note Model - Lưu metadata và kết quả xử lý AI
    """
    __tablename__ = 'notes'
    __table_args__ = (
        UniqueConstraint('user_id', 'note_id', name='uq_notes_user_note_id'),
    )
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PGUUID(as_uuid=True), ForeignKey('users.id'), nullable=False, index=True)
    
    # Metadata
    note_id = Column(String(100), nullable=False, index=True)  
    file_type = Column(String(50), nullable=True) 
    filename = Column(String(255), nullable=True)  
    file_size = Column(Integer, nullable=True)  
    
    # Processed content
    raw_text = Column(Text, nullable=True)  
    processed_text = Column(Text, nullable=True)  
    
    # AI Results
    summary = Column(Text, nullable=True)  
    summaries = Column(JSON, nullable=True)  
    questions = Column(JSON, nullable=True)  
    mcqs = Column(JSON, nullable=True)  
    review = Column(JSON, nullable=True)  
    
    # Job tracking (cho async processing)
    job_id = Column(String(100), nullable=True, index=True)  
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)  
    
    # Relationship
    user = relationship("User", back_populates="notes")
    feedbacks = relationship("Feedback", back_populates="note", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Note(id={self.id}, note_id={self.note_id}, file_type={self.file_type})>"
    
    def to_dict(self):
        """
        Convert model thành dict để trả về API
        """
        return {
            'id': str(self.id),
            'note_id': self.note_id,
            'user_id': str(self.user_id),
            'file_type': self.file_type,
            'filename': self.filename,
            'file_size': self.file_size,
            'raw_text': self.raw_text,
            'processed_text': self.processed_text,
            'summary': self.summary,
            'summaries': self.summaries,
            'questions': self.questions,
            'mcqs': self.mcqs,
            'review': self.review,
            'job_id': self.job_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
        }


class Feedback(Base):
    """
    Feedback Model - Lưu đánh giá của người dùng về tóm tắt
    """
    __tablename__ = 'feedbacks'
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    note_id = Column(PGUUID(as_uuid=True), ForeignKey('notes.id'), nullable=False, index=True)
    user_id = Column(PGUUID(as_uuid=True), ForeignKey('users.id'), nullable=False, index=True)
    
    # Rating (1-5 stars)
    rating = Column(Integer, nullable=False)
    
    # Feedback content
    comment = Column(Text, nullable=True) 
    feedback_type = Column(String(50), nullable=True)  
    
    # What user liked/disliked
    liked_aspects = Column(JSON, nullable=True) 
    disliked_aspects = Column(JSON, nullable=True) 
    
    # Suggestions for improvement
    suggestions = Column(Text, nullable=True) 
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    note = relationship("Note", back_populates="feedbacks")
    user = relationship("User", back_populates="feedbacks")
    
    def __repr__(self):
        return f"<Feedback(id={self.id}, note_id={self.note_id}, rating={self.rating})>"
    
    def to_dict(self):
        """
        Convert model thành dict để trả về API
        """
        return {
            'id': str(self.id),
            'note_id': str(self.note_id),
            'user_id': str(self.user_id),
            'rating': self.rating,
            'comment': self.comment,
            'feedback_type': self.feedback_type,
            'liked_aspects': self.liked_aspects,
            'disliked_aspects': self.disliked_aspects,
            'suggestions': self.suggestions,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }

