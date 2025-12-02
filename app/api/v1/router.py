from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional, List
import uuid
import os
from datetime import datetime

from app.agents.orchestrator import process_text, process_file, process_combined_inputs
from app.services.job_service import job_service
from app.services.db_service import db_service
from app.services.feedback_service import feedback_service
from app.database.database import get_db
from app.database.models import User

router = APIRouter()


def get_user_uuid(db: Session, user_id: str) -> uuid.UUID:
    """
    Helper function để lấy user UUID từ user_id (có thể là UUID hoặc username)
    
    Args:
        db: Database session
        user_id: User ID (UUID) hoặc username
        
    Returns:
        User UUID
        
    Raises:
        HTTPException: Nếu user không tồn tại
    """
    try:
        return uuid.UUID(user_id)
    except ValueError:
        user = db.query(User).filter(User.username == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
        return user.id

# ============= Synchronous Endpoints (Quick processing) =============

@router.post("/summarize")
async def summarize_text_sync(
    note: str = Form(...),
    user_id: Optional[str] = Form(None),
    note_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    API endpoint để tóm tắt text input trực tiếp (synchronous)
    Workflow: Clean → Summarize → Return
    Dùng cho text ngắn, trả về kết quả ngay
    
    Optional params:
    - user_id: User ID để lưu vào database
    - note_id: Custom note ID từ app
    """
    result = await process_text(note, db=db, use_rag=True)
    
    if user_id and note_id:
        try:
            # Get or create user
            user = db_service.get_or_create_user(db, username=user_id)
            
            # Create or update note (upsert)
            db_service.create_note(
                db=db,
                user_id=str(user.id),
                note_id=note_id,
                file_type='text',
                raw_text=result.get('raw_text'),
                processed_text=result.get('processed_text') or result.get('raw_text'),
                summary=result.get('summary'),
                summaries=result.get('summaries'),
                questions=result.get('questions'),
                mcqs=result.get('mcqs'),
                review=result.get('review')
            )
        except Exception as e:
            # Log error nhưng không fail request
            print(f"Error saving to database: {e}")
    
    return result

@router.post("/process")
async def process_input_sync(
    file: UploadFile = File(None),
    text: str = Form(None),
    user_id: Optional[str] = Form(None),
    note_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    API endpoint chính để xử lý input (file hoặc text) - Synchronous
    Workflow đầy đủ theo diagram với CrewAI agents
    Trả về kết quả ngay - có thể block nếu file lớn
    
    Optional params:
    - user_id: User ID để lưu vào database
    - note_id: Custom note ID từ app
    """
    if file:
        result = await process_file(file, db=db, use_rag=True)
        
        # Lưu vào database nếu có user_id và note_id
        if user_id and note_id:
            try:
                # Get or create user
                user = db_service.get_or_create_user(db, username=user_id)
                
                # Get file size
                file_size = None
                if hasattr(file, 'size'):
                    file_size = file.size
                else:
                    # Read file to get size
                    contents = await file.read()
                    file_size = len(contents)
                    await file.seek(0)  # Reset file pointer
                
                # Detect file type
                from app.core.detector import detect_input_type
                import tempfile
                import os
                suffix = os.path.splitext(file.filename)[1]
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                contents = await file.read()
                tmp.write(contents)
                tmp.close()
                file_type = detect_input_type(tmp.name)
                os.remove(tmp.name)
                await file.seek(0)  # Reset file pointer
                
                # Create note
                db_service.create_note(
                    db=db,
                    user_id=str(user.id),
                    note_id=note_id,
                    file_type=file_type,
                    filename=file.filename,
                    file_size=file_size,
                    raw_text=result.get('raw_text'),
                    processed_text=result.get('processed_text') or result.get('raw_text'),
                    summary=result.get('summary'),
                    summaries=result.get('summaries'),
                    questions=result.get('questions'),
                    mcqs=result.get('mcqs'),
                    review=result.get('review')
                )
            except Exception as e:
                print(f"Error saving to database: {e}")
                
    elif text:
        result = await process_text(text, db=db, use_rag=True)
        
        if user_id and note_id:
            try:
                # Get or create user
                user = db_service.get_or_create_user(db, username=user_id)
                
                # Create note
                db_service.create_note(
                    db=db,
                    user_id=str(user.id),
                    note_id=note_id,
                    file_type='text',
                    raw_text=result.get('raw_text'),
                    processed_text=result.get('processed_text') or result.get('raw_text'),
                    summary=result.get('summary'),
                    summaries=result.get('summaries'),
                    questions=result.get('questions'),
                    mcqs=result.get('mcqs'),
                    review=result.get('review')
                )
            except Exception as e:
                print(f"Error saving to database: {e}")
    else:
        return {"error": "Cần cung cấp file hoặc text"}
    
    return result


@router.post("/process/combined")
async def process_combined_endpoint(
    text_note: Optional[str] = Form(None),
    files: List[UploadFile] = File(None),
    user_id: Optional[str] = Form(None),
    note_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Endpoint mới cho phép gửi đồng thời nhiều input (text + nhiều file) trong một request
    để tạo ra ghi chú hoàn chỉnh.
    """
    uploads = files or []
    has_text = text_note and text_note.strip()

    if not has_text and not uploads:
        raise HTTPException(status_code=400, detail="Cần cung cấp ít nhất text_note hoặc files")

    result = await process_combined_inputs(
        text_note=text_note,
        files=uploads,
        db=db,
        use_rag=True
    )

    if user_id and note_id:
        try:
            user = db_service.get_or_create_user(db, username=user_id)
            review_payload = result.get('review') or {}
            if result.get('sources'):
                review_payload = {
                    **review_payload,
                    'sources': result.get('sources')
                }

            db_service.create_note(
                db=db,
                user_id=str(user.id),
                note_id=note_id,
                file_type='combined',
                filename=None,
                file_size=None,
                raw_text=result.get('raw_text'),
                processed_text=result.get('processed_text'),
                summary=result.get('summary'),
                summaries=result.get('summaries'),
                questions=result.get('questions'),
                mcqs=result.get('mcqs'),
                review=review_payload
            )
        except Exception as e:
            print(f"Error saving combined note: {e}")

    return result

# ============= Asynchronous Endpoints (Background processing) =============

@router.post("/process/async")
async def process_input_async(
    file: UploadFile = File(None),
    text: str = Form(None),
    user_id: Optional[str] = Form(None),
    note_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    API endpoint async - Trả về job_id ngay lập tức
    File sẽ được xử lý trong background worker (Celery)
    
    Request:
    - file: UploadFile (image/audio/pdf/docx/txt)
    - text: String (direct text input)
    - user_id: User ID để lưu vào database (optional)
    - note_id: Custom note ID từ app (optional)
    
    Response:
    {
        "job_id": "uuid-string",
        "status": "pending",
        "message": "File đang được xử lý..."
    }
    
    Sau đó dùng GET /jobs/{job_id}/status để check progress
    """
    if file:
        result = await job_service.create_file_processing_job(file, user_id, note_id, db)
    elif text:
        result = await job_service.create_text_processing_job(text, user_id, note_id, db)
    else:
        raise HTTPException(status_code=400, detail="Cần cung cấp file hoặc text")
    
    return result

@router.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """
    Kiểm tra status và progress của job
    
    """
    status = job_service.get_job_status(job_id)
    return status

@router.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """
    Lấy kết quả của job đã hoàn thành
    
    """
    result = job_service.get_job_result(job_id)
    
    if result is None:
        # Check status để xem job có tồn tại không
        status = job_service.get_job_status(job_id)
        if status.get('status') == 'pending' or status.get('status') == 'processing':
            raise HTTPException(
                status_code=202,
                detail=f"Job đang xử lý, status: {status.get('status')}"
            )
        elif status.get('status') == 'failed':
            raise HTTPException(
                status_code=500,
                detail=status.get('error', 'Job failed')
            )
        else:
            raise HTTPException(
                status_code=404,
                detail="Job không tồn tại hoặc chưa hoàn thành"
            )
    
    return result


# ============= Database Endpoints (History & Search) =============

@router.get("/users/{user_id}/notes")
async def get_user_notes(
    user_id: str,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    file_type: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """
    Lấy danh sách notes của user (History)
    
    Query params:
    - limit: Số lượng notes tối đa (1-100, default: 50)
    - offset: Offset cho pagination (default: 0)
    - file_type: Lọc theo loại file (optional: 'text', 'image', 'audio', 'pdf', 'docx')
    }
    """
    from app.database.models import Note
    
    # Get user UUID (support both UUID and username)
    user_uuid = get_user_uuid(db, user_id)
    
    notes = db_service.get_user_notes(
        db=db,
        user_id=user_id,
        limit=limit,
        offset=offset,
        file_type=file_type
    )
    
    # Count total
    total_query = db.query(Note).filter(Note.user_id == user_uuid)
    if file_type:
        total_query = total_query.filter(Note.file_type == file_type)
    total = total_query.count()
    
    return {
        "notes": [note.to_dict() for note in notes],
        "total": total,
        "limit": limit,
        "offset": offset
    }


@router.get("/users/{user_id}/notes/search")
async def search_user_notes(
    user_id: str,
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    Tìm kiếm notes của user
    
    Query params:
    - q: Text để tìm kiếm (required, min 1 char)
    - limit: Số lượng kết quả tối đa (1-100, default: 50)
    - offset: Offset cho pagination (default: 0)
    }
    """
    notes = db_service.search_notes(
        db=db,
        user_id=user_id,
        query_text=q,
        limit=limit,
        offset=offset
    )
    
    return {
        "notes": [note.to_dict() for note in notes],
        "total": len(notes), 
        "query": q,
        "limit": limit,
        "offset": offset
    }


@router.get("/notes/{note_id}")
async def get_note(
    note_id: str,
    user_id: Optional[str] = Query(None, description="User ID để lấy note (optional, nếu có thì tìm theo user_id + note_id)"),
    db: Session = Depends(get_db)
):
    """
    Lấy chi tiết một note theo ID
    
    Query params:
    - user_id: User ID (optional, nếu có thì tìm theo user_id + note_id để chính xác hơn)
    }
    """
    if user_id:
        note = db_service.get_note_by_user_and_note_id(db, user_id, note_id)
    else:
        note = db_service.get_note_by_id(db, note_id)
    
    if not note:
        raise HTTPException(status_code=404, detail="Note không tồn tại")
    
    return note.to_dict()


@router.delete("/notes/{note_id}")
async def delete_note(
    note_id: str,
    db: Session = Depends(get_db)
):
    """
    Xóa một note
    """
    success = db_service.delete_note(db, note_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Note không tồn tại")
    
    return {
        "success": True,
        "message": "Note đã được xóa"
    }


# ============= Feedback Endpoints =============

@router.post("/notes/{note_id}/feedback")
async def submit_feedback(
    note_id: str,
    rating: int = Form(..., ge=1, le=5, description="Rating từ 1-5 stars"),
    user_id: str = Form(...),
    comment: Optional[str] = Form(None),
    liked_aspects: Optional[str] = Form(None, description="JSON array of liked aspects"),
    disliked_aspects: Optional[str] = Form(None, description="JSON array of disliked aspects"),
    suggestions: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Submit feedback cho một note
    
    Request:
    - rating: Rating từ 1-5 stars (required)
    - user_id: User ID (required)
    - comment: Comment từ user (optional)
    - liked_aspects: JSON array of aspects user liked (optional)
    - disliked_aspects: JSON array of aspects user disliked (optional)
    - suggestions: Suggestions từ user (optional)
    }
    """
    import json
    
    # Parse JSON arrays nếu có
    liked = None
    disliked = None
    if liked_aspects:
        try:
            liked = json.loads(liked_aspects)
        except:
            liked = [liked_aspects] if liked_aspects else None
    
    if disliked_aspects:
        try:
            disliked = json.loads(disliked_aspects)
        except:
            disliked = [disliked_aspects] if disliked_aspects else None
    
    feedback = feedback_service.create_feedback(
        db=db,
        note_id=note_id,
        user_id=user_id,
        rating=rating,
        comment=comment,
        liked_aspects=liked,
        disliked_aspects=disliked,
        suggestions=suggestions
    )
    
    return feedback.to_dict()


@router.get("/notes/{note_id}/feedback")
async def get_note_feedbacks(
    note_id: str,
    db: Session = Depends(get_db)
):
    """
    Lấy tất cả feedbacks của một note
    """
    feedbacks = feedback_service.get_feedbacks_by_note(db, note_id)
    statistics = feedback_service.get_feedback_statistics(db, note_id=note_id)
    
    return {
        "feedbacks": [fb.to_dict() for fb in feedbacks],
        "statistics": statistics
    }


@router.get("/users/{user_id}/feedbacks")
async def get_user_feedbacks(
    user_id: str,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    Lấy danh sách feedbacks của user
    """
    feedbacks = feedback_service.get_user_feedbacks(
        db=db,
        user_id=user_id,
        limit=limit,
        offset=offset
    )
    
    return {
        "feedbacks": [fb.to_dict() for fb in feedbacks],
        "total": len(feedbacks),
        "limit": limit,
        "offset": offset
    }


@router.get("/feedback/statistics")
async def get_feedback_statistics(
    note_id: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """
    Lấy thống kê feedback
    
    Query params:
    - note_id: Note ID (optional, nếu None thì lấy tất cả)
    """
    statistics = feedback_service.get_feedback_statistics(db, note_id=note_id)
    return statistics


@router.get("/feedback/insights")
async def get_improvement_insights(
    limit: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db)
):
    """
    Lấy insights từ feedback để cải thiện prompts (RAG)
    
    Query params:
    - limit: Số lượng examples (1-20, default: 5)
    """
    insights = feedback_service.get_improvement_insights(db, limit=limit)
    return insights


@router.post("/notes/{note_id}/sync-result")
async def sync_note_result(
    note_id: str,
    db: Session = Depends(get_db)
):
    """
    Đồng bộ kết quả từ job vào note (cho các job đã hoàn thành nhưng chưa được update)
    """
    note = db_service.get_note_by_id(db, note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note không tồn tại")
    
    if not note.job_id:
        raise HTTPException(status_code=400, detail="Note không có job_id")
 
    job_status = job_service.get_job_status(note.job_id)
    
    if job_status.get('status') != 'completed':
        return {
            "success": False,
            "message": f"Job chưa hoàn thành. Status: {job_status.get('status')}",
            "job_status": job_status
        }

    job_result = job_service.get_job_result(note.job_id)
    if not job_result:
        raise HTTPException(status_code=404, detail="Không tìm thấy kết quả của job")
    
    processed_text = job_result.get('processed_text') or job_result.get('raw_text')
    
    updated_note = db_service.update_note(
        db=db,
        note_id=note_id,
        raw_text=job_result.get('raw_text'),
        processed_text=processed_text,
        summary=job_result.get('summary'),
        summaries=job_result.get('summaries'),
        questions=job_result.get('questions'),
        mcqs=job_result.get('mcqs'),
        review=job_result.get('review'),
        processed_at=datetime.utcnow()
    )
    
    if not updated_note:
        raise HTTPException(status_code=500, detail="Không thể cập nhật note")
    
    return {
        "success": True,
        "message": "Note đã được cập nhật với kết quả từ job",
        "note": updated_note.to_dict()
    }


# ============= Debug Endpoints =============

@router.post("/debug/test-ocr")
async def debug_test_ocr(
    file: UploadFile = File(...)
):
    """
    Debug endpoint để test OCR trực tiếp
    Giúp kiểm tra Tesseract có hoạt động không và xem lỗi chi tiết
    
    Request:
    - file: UploadFile (image)
    """
    import tempfile
    import os
    from app.core.preprocessor import process_image_file, configure_tesseract
    import pytesseract
    from PIL import Image
    
    # Save file to temp location
    suffix = os.path.splitext(file.filename)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    contents = await file.read()
    tmp.write(contents)
    tmp.close()
    file_path = tmp.name
    
    try:
        configure_tesseract()
        
        tesseract_version = None
        tesseract_error = None
        configured_path = None
        try:
            if hasattr(pytesseract.pytesseract, 'tesseract_cmd'):
                configured_path = pytesseract.pytesseract.tesseract_cmd
            tesseract_version = str(pytesseract.get_tesseract_version())
        except Exception as e:
            tesseract_error = str(e)
        
        image_info = None
        try:
            img = Image.open(file_path)
            image_info = {
                "size": img.size,
                "format": img.format,
                "mode": img.mode
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Không thể mở file ảnh: {e}",
                "tesseract_version": tesseract_version,
                "tesseract_error": tesseract_error,
                "tesseract_configured_path": configured_path
            }
        
        # Test OCR
        text, error_message = process_image_file(file_path)
        
        result = {
            "success": text != '',
            "text": text,
            "text_length": len(text),
            "tesseract_version": tesseract_version,
            "tesseract_error": tesseract_error,
            "tesseract_configured_path": configured_path,
            "image_info": image_info
        }
        
        if error_message:
            result["error"] = error_message
        
        return result
        
    finally:
        # Cleanup
        try:
            os.remove(file_path)
        except:
            pass


@router.get("/debug/celery-status")
async def debug_celery_status():
    """
    Debug endpoint để kiểm tra Celery worker và Redis connection
    
    """
    from app.services.celery_app import celery_app
    import redis
    
    result = {
        "redis_url": os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
        "redis_connected": False,
        "celery_workers": [],
        "active_tasks": [],
        "pending_tasks": 0,
        "errors": []
    }
    
    # Test Redis connection
    try:
        redis_client = redis.from_url(result["redis_url"])
        redis_client.ping()
        result["redis_connected"] = True
    except Exception as e:
        result["errors"].append(f"Redis connection error: {e}")
        return result
    
    # Inspect workers
    try:
        inspect = celery_app.control.inspect()
        
        # Get active workers
        active_workers = inspect.active()
        if active_workers:
            result["celery_workers"] = list(active_workers.keys())
            # Get active tasks
            for worker_name, tasks in active_workers.items():
                for task in tasks:
                    result["active_tasks"].append({
                        "worker": worker_name,
                        "task_id": task.get("id"),
                        "name": task.get("name"),
                        "args": task.get("args", []),
                    })
        else:
            result["errors"].append("No active Celery workers found!")
        
        # Get registered workers
        registered = inspect.registered()
        if registered:
            result["registered_workers"] = list(registered.keys())
        
        # Get stats
        stats = inspect.stats()
        if stats:
            result["worker_stats"] = stats
        
        # Count pending tasks (approximate)
        try:
            # Get queue length from Redis
            queue_key = celery_app.conf.task_default_queue or 'celery'
            pending_count = redis_client.llen(queue_key)
            result["pending_tasks"] = pending_count
        except:
            pass
            
    except Exception as e:
        result["errors"].append(f"Celery inspect error: {e}")
    
    return result
