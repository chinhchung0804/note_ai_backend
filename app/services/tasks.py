"""
Celery Tasks - Background workers for async processing
"""
import os
import tempfile
import asyncio
from typing import Optional
from celery import Task
from app.services.celery_app import celery_app
from app.core.detector import detect_input_type
from app.core.preprocessor import (
    process_image_file,
    process_audio_file,
    process_pdf_file,
    process_docx_file,
    clean_text
)
from app.agents.summarizer_agent import generate_learning_assets
from app.agents.reviewer_agent import review_text
from app.agents.ocr_agent import process_ocr_text
from app.agents.text_agent import process_and_normalize_text
from app.core.output_builder import build_output


class CallbackTask(Task):
    """Custom Task class với progress updates"""
    def on_success(self, retval, task_id, args, kwargs):
        """Khi task thành công"""
        update_task_state(task_id, 'SUCCESS', {
            'status': 'completed',
            'result': retval
        })

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Khi task fail"""
        update_task_state(task_id, 'FAILURE', {
            'status': 'failed',
            'error': str(exc)
        })


@celery_app.task(bind=True, base=CallbackTask, name='process_file_async')
def process_file_async(
    self,
    file_path: str,
    file_type: str,
    filename: str,
    user_id: Optional[str] = None,
    note_id: Optional[str] = None,
    note_db_id: Optional[str] = None
):
    """
    Process file async - Celery task
    
    Args:
        file_path: Path to file
        file_type: Type of file (image/audio/pdf/docx/text)
        filename: Original filename
        
    Returns:
        Dict với summary, review, raw_text
    """
    try:
        # Update progress: Started
        self.update_state(state='PROCESSING', meta={
            'status': 'processing',
            'progress': 10,
            'stage': 'Extracting text from file...'
        })
        
        # Step 1: Extract text từ file
        raw_text = ''
        error_message = ''
        print(f"📂 Đang xử lý file: {file_path}, type: {file_type}")
        
        if file_type == 'image':
            print("🖼️  Bắt đầu OCR cho file ảnh...")
            raw_text, error_message = process_image_file(file_path)
        elif file_type == 'audio':
            print("🎵 Bắt đầu transcribe audio...")
            raw_text = process_audio_file(file_path)
        elif file_type == 'pdf':
            print("📄 Bắt đầu extract text từ PDF...")
            raw_text = process_pdf_file(file_path)
        elif file_type == 'docx':
            print("📝 Bắt đầu extract text từ DOCX...")
            raw_text = process_docx_file(file_path)
        else:
            print("📄 Đang đọc file text...")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()
        
        print(f"📊 Text sau khi extract (trước clean): độ dài = {len(raw_text)} ký tự")
        raw_text = clean_text(raw_text)
        print(f"📊 Text sau khi clean: độ dài = {len(raw_text)} ký tự")
        
        if not raw_text or raw_text.strip() == '':
            if error_message:
                error_msg = f'Không thể extract text từ file {file_type}. Chi tiết lỗi: {error_message}'
            else:
                error_msg = (
                    f'Không thể extract text từ file {file_type}. '
                    f'Có thể do:\n'
                    f'1. File không có text (ảnh trống, audio không có giọng nói, v.v.)\n'
                    f'2. Chất lượng file kém (ảnh mờ, audio nhiễu)\n'
                    f'3. Tesseract OCR chưa được cài đặt (cho file ảnh)\n'
                    f'4. File bị hỏng hoặc format không hỗ trợ\n'
                    f'Vui lòng kiểm tra log trên để biết chi tiết.'
                )
            print(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        self.update_state(state='PROCESSING', meta={
            'status': 'processing',
            'progress': 30,
            'stage': 'Improving text quality...'
        })
        
        processed_text = raw_text
        review_result = None
        
        if file_type == 'image' or file_type == 'audio':
            # CrewAI Chain: OCR Agent → Text Agent → Reviewer Agent
            improved_text = asyncio.run(process_ocr_text(raw_text))
            normalized_text = asyncio.run(process_and_normalize_text(improved_text))
            review_result = asyncio.run(review_text(normalized_text, raw_text))
            processed_text = normalized_text
        else:
            # Text Agent → Reviewer Agent
            normalized_text = asyncio.run(process_and_normalize_text(raw_text))
            review_result = asyncio.run(review_text(normalized_text, raw_text))
            processed_text = normalized_text
        
        # Update progress: Text improved
        self.update_state(state='PROCESSING', meta={
            'status': 'processing',
            'progress': 60,
            'stage': 'Generating summary...'
        })
        
        try:
            from app.database.database import SessionLocal
            db = SessionLocal()
            try:
                learning_assets = asyncio.run(generate_learning_assets(
                    raw_text=processed_text,
                    db=db,
                    file_type=file_type,
                    use_rag=True
                ))
            finally:
                db.close()
        except Exception as exc:
            print(f"⚠️  Falling back to non-RAG learning assets: {exc}")
            learning_assets = asyncio.run(generate_learning_assets(
                raw_text=processed_text,
                db=None,
                file_type=file_type,
                use_rag=False
            ))
        
        # Update progress: Summary done
        self.update_state(state='PROCESSING', meta={
            'status': 'processing',
            'progress': 80,
            'stage': 'Finalizing results...'
        })
        
        result = build_output(
            summaries=learning_assets.get('summaries'),
            review=review_result,
            raw_text=raw_text,
            processed_text=processed_text,
            questions=learning_assets.get('questions'),
            mcqs=learning_assets.get('mcqs')
        )
        
        if note_db_id:
            try:
                from app.database.database import SessionLocal
                from app.services.db_service import db_service
                from datetime import datetime
                
                db = SessionLocal()
                try:
                    updated_note = db_service.update_note(
                        db=db,
                        note_id=note_db_id,
                        raw_text=raw_text,
                        processed_text=processed_text,
                        summary=learning_assets.get('summaries', {}).get('short_paragraph') if isinstance(learning_assets.get('summaries'), dict) else learning_assets.get('summaries'),
                        summaries=learning_assets.get('summaries'),
                        review=review_result,
                        questions=learning_assets.get('questions'),
                        mcqs=learning_assets.get('mcqs'),
                        processed_at=datetime.utcnow()
                    )
                    if updated_note:
                        print(f"✅ Successfully updated note {note_db_id} in database")
                    else:
                        print(f"⚠️  Note {note_db_id} not found in database")
                finally:
                    db.close()
            except Exception as e:
                import traceback
                print(f"❌ Error saving to database: {e}")
                print(f"Traceback: {traceback.format_exc()}")
        
        try:
            os.remove(file_path)
        except:
            pass
        
        return result
        
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e) or 'No error message'}"
        print(f"ERROR in process_file_async: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        
        try:
            os.remove(file_path)
        except:
            pass
        
        raise Exception(error_msg) from e


@celery_app.task(bind=True, base=CallbackTask, name='process_text_async')
def process_text_async(
    self,
    text: str,
    user_id: Optional[str] = None,
    note_id: Optional[str] = None,
    note_db_id: Optional[str] = None
):
    """
    Process text async - Celery task
    
    Args:
        text: Text to process
        
    Returns:
        Dict với summary, review, raw_text
    """
    try:
        self.update_state(state='PROCESSING', meta={
            'status': 'processing',
            'progress': 10,
            'stage': 'Processing text...'
        })
        
        txt = clean_text(text)
        
        self.update_state(state='PROCESSING', meta={
            'status': 'processing',
            'progress': 50,
            'stage': 'Generating summary...'
        })
        
        try:
            from app.database.database import SessionLocal
            db = SessionLocal()
            try:
                learning_assets = asyncio.run(generate_learning_assets(
                    raw_text=txt,
                    db=db,
                    file_type='text',
                    use_rag=True
                ))
            finally:
                db.close()
        except Exception as exc:
            print(f"⚠️  Falling back to non-RAG learning assets: {exc}")
            learning_assets = asyncio.run(generate_learning_assets(
                raw_text=txt,
                db=None,
                file_type='text',
                use_rag=False
            ))
        
        self.update_state(state='PROCESSING', meta={
            'status': 'processing',
            'progress': 90,
            'stage': 'Finalizing...'
        })
        
        result = build_output(
            summaries=learning_assets.get('summaries'),
            review={'valid': True, 'notes': 'Text input trực tiếp'},
            raw_text=txt,
            processed_text=txt,
            questions=learning_assets.get('questions'),
            mcqs=learning_assets.get('mcqs')
        )
        
        if note_db_id:
            try:
                from app.database.database import SessionLocal
                from app.services.db_service import db_service
                from datetime import datetime
                
                db = SessionLocal()
                try:
                    updated_note = db_service.update_note(
                        db=db,
                        note_id=note_db_id,
                        raw_text=txt,
                        processed_text=txt,
                        summary=learning_assets.get('summaries', {}).get('short_paragraph') if isinstance(learning_assets.get('summaries'), dict) else learning_assets.get('summaries'),
                        summaries=learning_assets.get('summaries'),
                        review={'valid': True, 'notes': 'Text input trực tiếp'},
                        questions=learning_assets.get('questions'),
                        mcqs=learning_assets.get('mcqs'),
                        processed_at=datetime.utcnow()
                    )
                    if updated_note:
                        print(f"✅ Successfully updated note {note_db_id} in database")
                    else:
                        print(f"⚠️  Note {note_db_id} not found in database")
                finally:
                    db.close()
            except Exception as e:
                import traceback
                print(f"❌ Error saving to database: {e}")
                print(f"Traceback: {traceback.format_exc()}")
        
        return result
        
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e) or 'No error message'}"
        print(f"ERROR in process_text_async: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        
        raise Exception(error_msg) from e


def update_task_state(task_id: str, state: str, meta: dict):
    """Helper function để update task state"""
    from app.services.celery_app import celery_app
    celery_app.backend.store_result(task_id, result=None, state=state, meta=meta)

