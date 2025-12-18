"""
Database Service - CRUD operations cho User và Note
"""
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime
import uuid
from app.database.models import User, Note


class DatabaseService:
    """
    Service để quản lý database operations
    """
    
    @staticmethod
    def get_or_create_user(db: Session, username: str, email: Optional[str] = None) -> User:
        """
        Lấy hoặc tạo user mới
        
        Args:
            db: Database session
            username: Username
            email: Email (optional)
            
        Returns:
            User object
        """
        user = db.query(User).filter(User.username == username).first()
        if not user:
            user = User(
                id=uuid.uuid4(),
                username=username,
                email=email
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        return user
    
    @staticmethod
    def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
        """
        Lấy user theo ID
        """
        try:
            return db.query(User).filter(User.id == uuid.UUID(user_id)).first()
        except:
            return None
    
    @staticmethod
    def get_note_by_user_and_note_id(db: Session, user_id: str, note_id: str) -> Optional[Note]:
        """
        Tìm note theo user_id và note_id (custom note_id)
        
        Args:
            db: Database session
            user_id: User ID (UUID string)
            note_id: Custom note ID từ app
            
        Returns:
            Note object hoặc None
        """
        try:
            user_uuid = uuid.UUID(user_id)
            return db.query(Note).filter(
                Note.user_id == user_uuid,
                Note.note_id == note_id
            ).first()
        except:
            return None

    @staticmethod
    def get_note_by_note_id(db: Session, note_id: str) -> Optional[Note]:
        """
        Tìm note chỉ theo note_id (dùng khi note_id có unique constraint toàn bảng)
        """
        try:
            return db.query(Note).filter(Note.note_id == note_id).first()
        except:
            return None
    
    @staticmethod
    def get_or_create_note(
        db: Session,
        user_id: str,
        note_id: str,
        file_type: Optional[str] = None,
        filename: Optional[str] = None,
        file_size: Optional[int] = None,
        job_id: Optional[str] = None
    ) -> Note:
        """
        Lấy hoặc tạo note mới
        - Nếu note với user_id + note_id đã tồn tại → UPDATE note đó (cập nhật file mới, job_id mới)
        - Nếu chưa tồn tại → Tạo note mới
        
        Args:
            db: Database session
            user_id: User ID (UUID string)
            note_id: Custom note ID từ app
            file_type: Loại file ('text', 'image', 'audio', 'pdf', 'docx')
            filename: Tên file
            file_size: Kích thước file
            job_id: Celery job ID (nếu async)
            
        Returns:
            Note object (existing hoặc newly created)
        """
        existing_note = DatabaseService.get_note_by_user_and_note_id(db, user_id, note_id)

        if existing_note:
            # Chỉ cập nhật metadata, không đổi user_id hay dùng fallback note_id chung
            if file_type is not None:
                existing_note.file_type = file_type
            if filename is not None:
                existing_note.filename = filename
            if file_size is not None:
                existing_note.file_size = file_size
            if job_id is not None:
                existing_note.job_id = job_id
            existing_note.raw_text = None
            existing_note.processed_text = None
            existing_note.summary = None
            existing_note.summaries = None
            existing_note.questions = None
            existing_note.mcqs = None
            existing_note.review = None
            existing_note.processed_at = None
            db.commit()
            db.refresh(existing_note)
            return existing_note

        note = Note(
            id=uuid.uuid4(),
            user_id=uuid.UUID(user_id),
            note_id=note_id,
            file_type=file_type,
            filename=filename,
            file_size=file_size,
            raw_text=None,
            processed_text=None,
            summary=None,
            review=None,
            job_id=job_id
        )
        db.add(note)
        db.commit()
        db.refresh(note)
        return note
    
    @staticmethod
    def create_note(
        db: Session,
        user_id: str,
        note_id: str,
        file_type: Optional[str] = None,
        filename: Optional[str] = None,
        file_size: Optional[int] = None,
        raw_text: Optional[str] = None,
        processed_text: Optional[str] = None,
        summary: Optional[str] = None,
        summaries: Optional[Dict[str, Any]] = None,
        questions: Optional[List[Dict[str, Any]]] = None,
        mcqs: Optional[Dict[str, Any]] = None,
        review: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None
    ) -> Note:
        """
        Tạo hoặc cập nhật note (upsert)
        - Nếu note với user_id + note_id đã tồn tại → UPDATE với dữ liệu mới
        - Nếu chưa tồn tại → Tạo note mới
        
        Args:
            db: Database session
            user_id: User ID
            note_id: Custom note ID từ app
            file_type: Loại file ('text', 'image', 'audio', 'pdf', 'docx', 'combined')
            filename: Tên file
            file_size: Kích thước file
            raw_text: Text gốc
            processed_text: Text đã xử lý
            summary: AI summary (đoạn chính)
            summaries: Full summary bundle (1 câu, 3-5 câu, bullet points)
            questions: Question set (5-10 câu)
            mcqs: MCQ set
            review: Review từ Reviewer Agent
            job_id: Celery job ID (nếu async)
            
        Returns:
            Note object (existing hoặc newly created)
        """
        existing_note = DatabaseService.get_note_by_user_and_note_id(db, user_id, note_id)
        
        if existing_note:
            if file_type is not None:
                existing_note.file_type = file_type
            if filename is not None:
                existing_note.filename = filename
            if file_size is not None:
                existing_note.file_size = file_size
            if raw_text is not None:
                existing_note.raw_text = raw_text
            if processed_text is not None:
                existing_note.processed_text = processed_text
            if summary is not None:
                existing_note.summary = summary
            if summaries is not None:
                existing_note.summaries = summaries
            if questions is not None:
                existing_note.questions = questions
            if mcqs is not None:
                existing_note.mcqs = mcqs
            if review is not None:
                existing_note.review = review
            if job_id is not None:
                existing_note.job_id = job_id
            if processed_text is not None or summary is not None:
                existing_note.processed_at = datetime.utcnow()
            existing_note.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(existing_note)
            return existing_note
        else:
            note = Note(
                id=uuid.uuid4(),
                user_id=uuid.UUID(user_id),
                note_id=note_id,
                file_type=file_type,
                filename=filename,
                file_size=file_size,
                raw_text=raw_text,
                processed_text=processed_text,
                summary=summary,
                summaries=summaries,
                questions=questions,
                mcqs=mcqs,
                review=review,
                job_id=job_id
            )
            db.add(note)
            db.commit()
            db.refresh(note)
            return note
    
    @staticmethod
    def update_note(
        db: Session,
        note_id: str,
        processed_text: Optional[str] = None,
        summary: Optional[str] = None,
        summaries: Optional[Dict[str, Any]] = None,
        questions: Optional[List[Dict[str, Any]]] = None,
        mcqs: Optional[Dict[str, Any]] = None,
        review: Optional[Dict[str, Any]] = None,
        processed_at: Optional[datetime] = None,
        job_id: Optional[str] = None,
        raw_text: Optional[str] = None
    ) -> Optional[Note]:
        """
        Cập nhật note sau khi xử lý xong
        
        Args:
            db: Database session
            note_id: Note ID (custom note_id hoặc UUID)
            processed_text: Text đã xử lý
            summary: AI summary (đoạn 3-5 câu chính)
            summaries: Full summary bundle
            questions: Question set (5-10 câu)
            mcqs: MCQ set theo độ khó
            review: Review từ Reviewer Agent
            processed_at: Thời gian xử lý xong
            job_id: Celery job ID (để update job_id)
            raw_text: Raw text (để update raw_text)
            
        Returns:
            Updated Note object hoặc None
        """
        from sqlalchemy import or_
        try:
            note = db.query(Note).filter(
                or_(
                    Note.note_id == note_id,
                    Note.id == uuid.UUID(note_id)
                )
            ).first()
        except:
            note = db.query(Note).filter(Note.note_id == note_id).first()
        
        if not note:
            return None
        
        if processed_text is not None:
            note.processed_text = processed_text
        if summary is not None:
            note.summary = summary
        if summaries is not None:
            note.summaries = summaries
        if questions is not None:
            note.questions = questions
        if mcqs is not None:
            note.mcqs = mcqs
        if review is not None:
            note.review = review
        if raw_text is not None:
            note.raw_text = raw_text
        if job_id is not None:
            note.job_id = job_id
        if processed_at is not None:
            note.processed_at = processed_at
        elif processed_text is not None or summary is not None:
            # Auto set processed_at nếu đang update kết quả xử lý
            note.processed_at = datetime.utcnow()
        
        db.commit()
        db.refresh(note)
        return note
    
    @staticmethod
    def get_note_by_id(db: Session, note_id: str) -> Optional[Note]:
        """
        Lấy note theo ID (custom note_id hoặc UUID)
        """
        from sqlalchemy import or_
        try:
            return db.query(Note).filter(
                or_(
                    Note.note_id == note_id,
                    Note.id == uuid.UUID(note_id)
                )
            ).first()
        except:
            return db.query(Note).filter(Note.note_id == note_id).first()
    
    @staticmethod
    def get_note_by_job_id(db: Session, job_id: str) -> Optional[Note]:
        """
        Lấy note theo Celery job_id
        """
        return db.query(Note).filter(Note.job_id == job_id).first()
    
    @staticmethod
    def get_user_notes(
        db: Session,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        file_type: Optional[str] = None
    ) -> List[Note]:
        """
        Lấy danh sách notes của user (History)
        
        Args:
            db: Database session
            user_id: User ID (UUID) hoặc username
            limit: Số lượng notes tối đa
            offset: Offset cho pagination
            file_type: Lọc theo loại file (optional)
            
        Returns:
            List of Note objects
        """
        from app.database.models import Note
        from sqlalchemy import desc
        
        try:
            user_uuid = uuid.UUID(user_id)
        except ValueError:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return []  
            user_uuid = user.id
        
        query = db.query(Note).filter(Note.user_id == user_uuid)
        
        if file_type:
            query = query.filter(Note.file_type == file_type)
        
        return query.order_by(desc(Note.created_at)).limit(limit).offset(offset).all()
    
    @staticmethod
    def search_notes(
        db: Session,
        user_id: str,
        query_text: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Note]:
        """
        Tìm kiếm notes theo text (search trong summary, raw_text, processed_text)
        
        Args:
            db: Database session
            user_id: User ID (UUID) hoặc username
            query_text: Text để tìm kiếm
            limit: Số lượng kết quả tối đa
            offset: Offset cho pagination
            
        Returns:
            List of Note objects
        """
        from app.database.models import Note
        from sqlalchemy import desc, or_
        
        try:
            user_uuid = uuid.UUID(user_id)
        except ValueError:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return []  
            user_uuid = user.id
        
        search_pattern = f"%{query_text}%"
        
        return db.query(Note).filter(
            Note.user_id == user_uuid,
            or_(
                Note.summary.ilike(search_pattern),
                Note.raw_text.ilike(search_pattern),
                Note.processed_text.ilike(search_pattern)
            )
        ).order_by(desc(Note.created_at)).limit(limit).offset(offset).all()
    
    @staticmethod
    def delete_note(db: Session, note_id: str) -> bool:
        """
        Xóa note
        
        Args:
            db: Database session
            note_id: Note ID (custom note_id hoặc UUID)
            
        Returns:
            True nếu xóa thành công, False nếu không tìm thấy
        """
        note = DatabaseService.get_note_by_id(db, note_id)
        if not note:
            return False
        
        db.delete(note)
        db.commit()
        return True


# Global instance
db_service = DatabaseService()

