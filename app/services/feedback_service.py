"""
Feedback Service - Quản lý feedback và RAG cho prompt improvement
"""
from typing import Optional, List, Dict, Any
from sqlalchemy.exc import OperationalError, InterfaceError
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, and_
from datetime import datetime
import uuid
from app.database.models import Feedback, Note, User


class FeedbackService:
    """
    Service để quản lý feedback và cải thiện prompts
    """
    
    @staticmethod
    def create_feedback(
        db: Session,
        note_id: str,
        user_id: str,
        rating: int,
        comment: Optional[str] = None,
        feedback_type: Optional[str] = None,
        liked_aspects: Optional[List[str]] = None,
        disliked_aspects: Optional[List[str]] = None,
        suggestions: Optional[str] = None
    ) -> Feedback:
        """
        Tạo feedback mới
        
        Args:
            db: Database session
            note_id: Note ID (custom note_id hoặc UUID)
            user_id: User ID
            rating: Rating (1-5 stars)
            comment: Comment từ user (optional)
            feedback_type: 'positive', 'negative', 'neutral' (optional)
            liked_aspects: List of aspects user liked (optional)
            disliked_aspects: List of aspects user disliked (optional)
            suggestions: Suggestions từ user (optional)
            
        Returns:
            Feedback object
        """
        from app.services.db_service import db_service
        note = db_service.get_note_by_id(db, note_id)
        if not note:
            raise ValueError(f"Note không tồn tại: {note_id}")
        
        user = db_service.get_user_by_id(db, user_id)
        if not user:
            raise ValueError(f"User không tồn tại: {user_id}")
        
        # Validate rating
        if rating < 1 or rating > 5:
            raise ValueError("Rating phải từ 1-5")
        
        if not feedback_type:
            if rating >= 4:
                feedback_type = 'positive'
            elif rating <= 2:
                feedback_type = 'negative'
            else:
                feedback_type = 'neutral'
        
        feedback = Feedback(
            id=uuid.uuid4(),
            note_id=note.id,
            user_id=user.id,
            rating=rating,
            comment=comment,
            feedback_type=feedback_type,
            liked_aspects=liked_aspects,
            disliked_aspects=disliked_aspects,
            suggestions=suggestions
        )
        
        db.add(feedback)
        db.commit()
        db.refresh(feedback)
        
        return feedback
    
    @staticmethod
    def get_feedback_by_id(db: Session, feedback_id: str) -> Optional[Feedback]:
        """
        Lấy feedback theo ID
        """
        try:
            return db.query(Feedback).filter(Feedback.id == uuid.UUID(feedback_id)).first()
        except:
            return None
    
    @staticmethod
    def get_feedbacks_by_note(db: Session, note_id: str) -> List[Feedback]:
        """
        Lấy tất cả feedbacks của một note
        """
        from app.services.db_service import db_service
        note = db_service.get_note_by_id(db, note_id)
        if not note:
            return []
        
        return db.query(Feedback).filter(Feedback.note_id == note.id).order_by(desc(Feedback.created_at)).all()
    
    @staticmethod
    def get_user_feedbacks(
        db: Session,
        user_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Feedback]:
        """
        Lấy danh sách feedbacks của user
        """
        return db.query(Feedback).filter(
            Feedback.user_id == uuid.UUID(user_id)
        ).order_by(desc(Feedback.created_at)).limit(limit).offset(offset).all()
    
    @staticmethod
    def get_positive_feedbacks(
        db: Session,
        limit: int = 10
    ) -> List[Feedback]:
        """
        Lấy positive feedbacks để học hỏi (RAG)
        """
        return db.query(Feedback).filter(
            and_(
                Feedback.rating >= 4,
                Feedback.comment.isnot(None)
            )
        ).order_by(desc(Feedback.created_at)).limit(limit).all()
    
    @staticmethod
    def get_negative_feedbacks(
        db: Session,
        limit: int = 10
    ) -> List[Feedback]:
        """
        Lấy negative feedbacks để học hỏi (RAG)
        """
        return db.query(Feedback).filter(
            and_(
                Feedback.rating <= 2,
                Feedback.comment.isnot(None)
            )
        ).order_by(desc(Feedback.created_at)).limit(limit).all()
    
    @staticmethod
    def get_feedback_statistics(db: Session, note_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Lấy thống kê feedback
        
        Args:
            note_id: Note ID (optional, nếu None thì lấy tất cả)
            
        Returns:
            Dict với statistics
        """
        query = db.query(Feedback)
        
        if note_id:
            from app.services.db_service import db_service
            note = db_service.get_note_by_id(db, note_id)
            if note:
                query = query.filter(Feedback.note_id == note.id)
        
        total = query.count()
        
        if total == 0:
            return {
                'total': 0,
                'average_rating': 0,
                'rating_distribution': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }
        
        # Average rating
        avg_rating = db.query(func.avg(Feedback.rating)).filter(
            query.whereclause if hasattr(query, 'whereclause') else True
        ).scalar() or 0
        
        # Rating distribution
        rating_dist = {}
        for rating in range(1, 6):
            count = query.filter(Feedback.rating == rating).count()
            rating_dist[rating] = count
        
        # Feedback type counts
        positive_count = query.filter(Feedback.feedback_type == 'positive').count()
        negative_count = query.filter(Feedback.feedback_type == 'negative').count()
        neutral_count = query.filter(Feedback.feedback_type == 'neutral').count()
        
        return {
            'total': total,
            'average_rating': round(float(avg_rating), 2),
            'rating_distribution': rating_dist,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count
        }
    
    @staticmethod
    def get_improvement_insights(db: Session, limit: int = 5) -> Dict[str, Any]:
        """
        Lấy insights từ feedbacks để cải thiện prompts (RAG)
        
        Returns:
            Dict với insights:
            - positive_examples: Examples của good summaries
            - negative_examples: Examples của bad summaries
            - common_liked_aspects: Aspects users thường thích
            - common_disliked_aspects: Aspects users thường không thích
            - suggestions: Common suggestions
        """
        try:
            # Get positive feedbacks với notes
            positive_feedbacks = FeedbackService.get_positive_feedbacks(db, limit=limit)
            positive_examples = []
            for fb in positive_feedbacks:
                note = db.query(Note).filter(Note.id == fb.note_id).first()
                if note and note.summary:
                    positive_examples.append({
                        'summary': note.summary,
                        'raw_text': note.raw_text[:200] if note.raw_text else None,  # First 200 chars
                        'rating': fb.rating,
                        'comment': fb.comment,
                        'liked_aspects': fb.liked_aspects
                    })
            
            # Get negative feedbacks với notes
            negative_feedbacks = FeedbackService.get_negative_feedbacks(db, limit=limit)
            negative_examples = []
            for fb in negative_feedbacks:
                note = db.query(Note).filter(Note.id == fb.note_id).first()
                if note and note.summary:
                    negative_examples.append({
                        'summary': note.summary,
                        'raw_text': note.raw_text[:200] if note.raw_text else None,
                        'rating': fb.rating,
                        'comment': fb.comment,
                        'disliked_aspects': fb.disliked_aspects,
                        'suggestions': fb.suggestions
                    })
            
            # Common liked aspects
            all_liked = []
            for fb in positive_feedbacks:
                if fb.liked_aspects:
                    all_liked.extend(fb.liked_aspects)
            
            # Common disliked aspects
            all_disliked = []
            for fb in negative_feedbacks:
                if fb.disliked_aspects:
                    all_disliked.extend(fb.disliked_aspects)
            
            # Common suggestions
            all_suggestions = []
            for fb in negative_feedbacks:
                if fb.suggestions:
                    all_suggestions.append(fb.suggestions)
            
            return {
                'positive_examples': positive_examples,
                'negative_examples': negative_examples,
                'common_liked_aspects': list(set(all_liked))[:10],  # Top 10 unique
                'common_disliked_aspects': list(set(all_disliked))[:10],
                'suggestions': all_suggestions[:10]
            }
        except (OperationalError, InterfaceError) as e:
            print(f"[feedback_service] Skip improvement insights due to DB error: {e}")
            return {}


# Global instance
feedback_service = FeedbackService()

