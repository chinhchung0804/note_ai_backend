"""
Job Service - Quản lý background jobs
"""
import os
import tempfile
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from app.services.tasks import process_file_async, process_text_async
from app.services.db_service import db_service
from app.core.detector import detect_input_type


class JobService:
    """Service để quản lý background jobs"""
    
    @staticmethod
    async def create_file_processing_job(
        upload_file,
        user_id: Optional[str] = None,
        note_id: Optional[str] = None,
        db: Optional[Session] = None
    ) -> Dict[str, str]:
        """
        Tạo job để xử lý file async
        
        Args:
            upload_file: Uploaded file object
            user_id: User ID để lưu vào database (optional)
            note_id: Custom note ID từ app (optional)
            db: Database session (optional)
            
        Returns:
            Dict với job_id
        """
        # Save file to temp location
        suffix = os.path.splitext(upload_file.filename)[1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        contents = await upload_file.read()
        tmp.write(contents)
        tmp.flush()
        tmp.close()
        
        file_path = tmp.name
        input_type = detect_input_type(file_path)
        
        # Get file size
        file_size = len(contents)
        
        note_db_id = None
        if user_id and note_id and db:
            try:
                # Get or create user
                user = db_service.get_or_create_user(db, username=user_id)
                note = db_service.get_or_create_note(
                    db=db,
                    user_id=str(user.id),
                    note_id=note_id,
                    file_type=input_type,
                    filename=upload_file.filename,
                    file_size=file_size,
                    job_id=None 
                )
                note_db_id = str(note.id)
            except Exception as e:
                print(f"Error creating/updating note in database: {e}")
        
        # Submit task to Celery
        task = process_file_async.delay(
            file_path=file_path,
            file_type=input_type,
            filename=upload_file.filename,
            user_id=user_id,
            note_id=note_id,
            note_db_id=note_db_id
        )
        
        # Update note với job_id nếu có
        if note_db_id and db:
            try:
                db_service.update_note(
                    db=db,
                    note_id=note_db_id,
                    job_id=task.id
                )
            except Exception as e:
                print(f"Error updating note with job_id: {e}")
        
        return {
            'job_id': task.id,
            'status': 'pending',
            'message': 'File đang được xử lý...'
        }
    
    @staticmethod
    async def create_text_processing_job(
        text: str,
        user_id: Optional[str] = None,
        note_id: Optional[str] = None,
        db: Optional[Session] = None
    ) -> Dict[str, str]:
        """
        Tạo job để xử lý text async
        
        Args:
            text: Text to process
            user_id: User ID để lưu vào database (optional)
            note_id: Custom note ID từ app (optional)
            db: Database session (optional)
            
        Returns:
            Dict với job_id
        """
        note_db_id = None
        if user_id and note_id and db:
            try:
                # Get or create user
                user = db_service.get_or_create_user(db, username=user_id)
                
                note = db_service.get_or_create_note(
                    db=db,
                    user_id=str(user.id),
                    note_id=note_id,
                    file_type='text',
                    filename=None,
                    file_size=None,
                    job_id=None  
                )
                note_db_id = str(note.id)
            except Exception as e:
                print(f"Error creating/updating note in database: {e}")
        
        task = process_text_async.delay(
            text=text,
            user_id=user_id,
            note_id=note_id,
            note_db_id=note_db_id
        )
        
        if note_db_id and db:
            try:
                db_service.update_note(
                    db=db,
                    note_id=note_db_id,
                    job_id=task.id
                )
            except Exception as e:
                print(f"Error updating note with job_id: {e}")
        
        return {
            'job_id': task.id,
            'status': 'pending',
            'message': 'Text đang được xử lý...'
        }
    
    @staticmethod
    def get_job_status(job_id: str) -> Dict[str, Any]:
        """
        Lấy status của job
        
        Args:
            job_id: Job ID
            
        Returns:
            Dict với status và progress
        """
        from app.services.celery_app import celery_app
        
        try:
            task = celery_app.AsyncResult(job_id)
            
            if task.state == 'PENDING':
                return {
                    'job_id': job_id,
                    'status': 'pending',
                    'progress': 0,
                    'stage': 'Waiting to start...'
                }
            elif task.state == 'PROCESSING':
                return task.info
            elif task.state == 'SUCCESS':
                return {
                    'job_id': job_id,
                    'status': 'completed',
                    'progress': 100,
                    'stage': 'Completed',
                    'result': task.result
                }
            elif task.state == 'FAILURE':
                error_msg = 'Unknown error'
                if task.info:
                    if isinstance(task.info, Exception):
                        error_msg = str(task.info) or type(task.info).__name__
                    elif isinstance(task.info, (str, dict)):
                        error_msg = str(task.info)
                    else:
                        error_msg = repr(task.info)
                
                return {
                    'job_id': job_id,
                    'status': 'failed',
                    'progress': 0,
                    'error': error_msg if error_msg != 'None' else 'Task failed with unknown error. Check Celery worker logs for details.'
                }
            else:
                return {
                    'job_id': job_id,
                    'status': task.state.lower(),
                    'progress': 0
                }
                
        except Exception as e:
            return {
                'job_id': job_id,
                'status': 'error',
                'error': str(e)
            }
    
    @staticmethod
    def get_job_result(job_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy kết quả của job đã hoàn thành
        
        Args:
            job_id: Job ID
            
        Returns:
            Result dict hoặc None
        """
        from app.services.celery_app import celery_app
        
        try:
            task = celery_app.AsyncResult(job_id)
            
            if task.state == 'SUCCESS':
                return task.result
            else:
                return None
                
        except:
            return None


# Global instance
job_service = JobService()

