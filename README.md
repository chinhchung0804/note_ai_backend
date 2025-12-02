<<<<<<< HEAD
# note_ai_backend
=======
# Note Summarizer AI Backend (FastAPI + LangChain + CrewAI + Google Gemini)

## Overview
Backend for Note Summarizer mobile app (Android). Uses LangChain + CrewAI + Google Gemini for LLM tasks,
includes OCR (Tesseract), STT (Whisper), và hỗ trợ nhiều loại file.

### Supported File Types
- 📝 **Text** - Plain text files
- 🖼️ **Image** - JPG, PNG (với OCR)
- 🎵 **Audio** - MP3, WAV, etc. (với STT)
- 📄 **PDF** - Extract text từ PDF
- 📋 **DOCX** - Extract text từ Word documents

## Quickstart (local)
1. Copy `.env.example` to `.env` and fill `GOOGLE_API_KEY`.
2. Create virtualenv and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run dev server:
   ```bash
   uvicorn app.main:app --reload
   ```
4. API docs: http://localhost:8000/docs

## Docker (optional)
```bash
docker-compose up --build
```

## Project layout
- `app/` - main application code (api, agents, core, services)
  - `core/` - Input detector, preprocessor (OCR/STT/PDF/DOCX), output builder
  - `agents/` - AI agents (OCR, Text, Reviewer, Summarizer, Orchestrator)
- `tests/` - simple tests

## API Endpoints

### POST `/api/v1/process`
Process input (file hoặc text)
- `file`: UploadFile (image/audio/pdf/docx/txt)
- `text`: String (direct text input)

Response (rút gọn): 
```json
{
  "summary": "Tóm tắt 3-5 câu",
  "summaries": {
    "one_sentence": "...",
    "short_paragraph": "...",
    "bullet_points": ["..."]
  },
  "questions": [{"question": "...", "answer": "..."}],
  "mcqs": {"easy": [...], "medium": [...], "hard": [...]},
  "review": {"valid": true, "notes": "..."},
  "raw_text": "Text gốc...",
  "processed_text": "Text sau chuẩn hóa"
}
```

### POST `/api/v1/summarize`
Summarize text trực tiếp
- `note`: String

### POST `/api/v1/process/async` (Background Processing)
Submit task async, trả về `job_id` ngay lập tức
- `file`: UploadFile hoặc `text`: String
- **Response**: `{"job_id": "uuid", "status": "pending"}`

### GET `/api/v1/jobs/{job_id}/status`
Check status và progress của job
- **Response**: `{"status": "processing", "progress": 45, "stage": "..."}`

### GET `/api/v1/jobs/{job_id}/result`
Lấy kết quả khi job completed

> 📖 Xem chi tiết: [BACKGROUND_PROCESSING.md](BACKGROUND_PROCESSING.md)

## Tech Stack
- **FastAPI** - Web framework
- **LangChain** - LLM orchestration
- **CrewAI** - Multi-agent system (OCR Agent, Text Agent, Reviewer Agent)
- **Google Gemini** - LLM cho summarization
- **Tesseract OCR** - Image → Text
- **Whisper** - Audio → Text
- **PyPDF** - PDF text extraction
- **python-docx** - DOCX text extraction
- **Celery** - Background task queue
- **Redis** - Message broker & result backend
- **Docker Compose** - Container orchestration

## Features
- ✅ **Multi-input Support**: Text, Image, Audio, PDF, DOCX
- ✅ **Background Processing**: Really async với Celery + Redis
- ✅ **Real-time Progress**: Track status và progress
- ✅ **Scalable**: Multiple workers, không block API
- ✅ **Production Ready**: Docker, error handling, logging
- ✅ **Learning Assets**: Tự động sinh 3 kiểu tóm tắt + 5-10 câu hỏi ôn tập + MCQ 3 độ khó

## Database migrations
- Với database mới: `python -m app.database.init_db`
- Với database cũ cần bổ sung cột `summaries/questions/mcqs`: `python -m app.database.migrations`

## Testing

### Quick Test Script
Chạy script test nhanh để kiểm tra các chức năng chính:
```bash
python quick_test.py
```

**Yêu cầu trước khi chạy:**
1. ✅ Redis đang chạy: `redis-server`
2. ✅ Celery worker đang chạy: `celery -A app.services.celery_app.celery_app worker --loglevel=info`
3. ✅ FastAPI server đang chạy: `uvicorn app.main:app --reload`

### Manual Testing
Xem hướng dẫn chi tiết: [TESTING_GUIDE.md](TESTING_GUIDE.md)

### Unit Tests
```bash
pytest tests/test_summarizer.py
```
>>>>>>> 980fece (first commit)
