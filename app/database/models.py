"""
Database Models - User, Note, Feedback
"""
from datetime import datetime
import uuid

from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer, JSON
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship

from app.database.database import Base


class User(Base):
    """
    User Model - Lưu thông tin người dùng
    """
    __tablename__ = 'users'
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship với notes và feedbacks
    notes = relationship("Note", back_populates="user", cascade="all, delete-orphan")
    feedbacks = relationship("Feedback", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username})>"


class Note(Base):
    """
    Note Model - Lưu metadata và kết quả xử lý AI
    """
    __tablename__ = 'notes'
    
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
    job_id = Column(String(100), nullable=True, index=True)  # Celery job ID
    
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

