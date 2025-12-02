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

## 📋 Yêu cầu và Chuẩn bị

### 1. Cài đặt Python và Dependencies

#### 1.1. Python
- **Yêu cầu**: Python 3.11 trở lên
- Kiểm tra phiên bản: `python --version` hoặc `python3 --version`

#### 1.2. Tạo Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

#### 1.3. Cài đặt Python Packages
```bash
pip install -r requirements.txt
```

### 2. Cài đặt Tesseract OCR

Tesseract OCR cần thiết để xử lý ảnh (Image → Text).

#### Windows
1. Tải Tesseract từ: https://github.com/UB-Mannheim/tesseract/wiki
2. Cài đặt (khuyến nghị: `C:\Program Files\Tesseract-OCR`)
3. Thêm vào PATH hoặc cấu hình trong code:
   ```python
   # Nếu không có trong PATH, thêm vào .env:
   TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
   ```

#### Kiểm tra cài đặt
```bash
tesseract --version
```

### 3. Cài đặt Whisper STT (Speech-to-Text)

Whisper được cài đặt qua Python package `openai-whisper`, nhưng cần **FFmpeg** để xử lý audio.

#### Windows
1. Tải FFmpeg từ: https://ffmpeg.org/download.html
2. Giải nén và thêm vào PATH
3. Hoặc sử dụng Chocolatey: `choco install ffmpeg`

#### Kiểm tra cài đặt
```bash
ffmpeg -version
```

**Lưu ý**: Whisper model sẽ được tải tự động lần đầu sử dụng (khoảng 1.5GB).

### 4. Cài đặt Redis

Redis được sử dụng làm message broker và result backend cho Celery.

#### Cách 1: Sử dụng Docker (Khuyến nghị)
```bash
docker run -d -p 6379:6379 --name redis redis:7
```

#### Cách 2: Cài đặt trực tiếp

**Windows:**
- Tải từ: https://github.com/microsoftarchive/redis/releases
- Hoặc sử dụng WSL2

#### Kiểm tra Redis đang chạy
```bash
redis-cli ping
# Kết quả mong đợi: PONG
```

### 5. Cài đặt PostgreSQL

PostgreSQL được sử dụng để lưu trữ dữ liệu.

#### Cách 1: Sử dụng Docker (Khuyến nghị)
```bash
docker run -d \
  --name postgres \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=note_ai \
  -p 5432:5432 \
  postgres:16
```

#### Cách 2: Cài đặt trực tiếp

**Windows:**
- Tải từ: https://www.postgresql.org/download/windows/
- Hoặc sử dụng WSL2

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```
#### Tạo Database
```bash
# Kết nối PostgreSQL
psql -U postgres

# Tạo database
CREATE DATABASE note_ai;
CREATE USER user WITH PASSWORD 'password';
GRANT ALL PRIVILEGES ON DATABASE note_ai TO user;
\q
```

**Lưu ý**: 
- Lấy `GOOGLE_API_KEY` từ: https://makersuite.google.com/app/apikey
- Đảm bảo `.env` không được commit lên Git (đã có trong `.gitignore`)

### 7. Khởi tạo Database

Sau khi PostgreSQL đã chạy, khởi tạo database schema:

```bash
python -m app.database.init_db
```

Nếu database đã tồn tại và cần migrate:
```bash
python -m app.database.migrations
```

## 🚀 Chạy Backend

### Cách 1: Chạy Local (Development)

#### Bước 1: Khởi động Redis
```bash
# Nếu dùng Docker
docker start redis

# Hoặc nếu cài đặt trực tiếp
redis-server
```

#### Bước 2: Khởi động PostgreSQL
```bash
# Nếu dùng Docker
docker start postgres

```

#### Bước 3: Khởi động Celery Worker
Mở terminal mới và chạy:

**Windows:**
```bash
# Sử dụng script có sẵn
run_worker.bat

# Hoặc chạy trực tiếp
celery -A app.services.celery_app worker --loglevel=info --pool=solo
```

#### Bước 4: Khởi động FastAPI Server
Mở terminal mới và chạy:
```bash
# Đảm bảo virtual environment đã được activate
uvicorn app.main:app --reload
```

#### Bước 5: Kiểm tra
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Cách 2: Chạy với Docker Compose (Production-like)

Tất cả services (Backend, Celery, Redis, PostgreSQL) sẽ được khởi động tự động:

```bash
docker-compose up --build
```

**Lưu ý**: 
- Đảm bảo file `.env` đã được tạo và cấu hình đúng
- Lần đầu chạy có thể mất vài phút để build images

Để chạy ở background:
```bash
docker-compose up -d
```

Xem logs:
```bash
docker-compose logs -f
```

Dừng services:
```bash
docker-compose down
```

## 📁 Project Layout
- `app/` - main application code (api, agents, core, services)
  - `core/` - Input detector, preprocessor (OCR/STT/PDF/DOCX), output builder
  - `agents/` - AI agents (OCR, Text, Reviewer, Summarizer, Orchestrator)
  - `services/` - Celery tasks, database service, storage service
  - `database/` - Database models, migrations, initialization
- `tests/` - Unit tests và integration tests

## 🔌 API Endpoints

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

> 📖 Xem chi tiết: [TESTING_GUIDE.md](TESTING_GUIDE.md)

## 🛠️ Tech Stack
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
- **PostgreSQL** - Database
- **Docker Compose** - Container orchestration

## ✨ Features
- ✅ **Multi-input Support**: Text, Image, Audio, PDF, DOCX
- ✅ **Background Processing**: Really async với Celery + Redis
- ✅ **Real-time Progress**: Track status và progress
- ✅ **Scalable**: Multiple workers, không block API
- ✅ **Production Ready**: Docker, error handling, logging
- ✅ **Learning Assets**: Tự động sinh 3 kiểu tóm tắt + 5-10 câu hỏi ôn tập + MCQ 3 độ khó

## 🗄️ Database Migrations
- Với database mới: `python -m app.database.init_db`
- Với database cũ cần bổ sung cột `summaries/questions/mcqs`: `python -m app.database.migrations`

## 🧪 Testing

### Quick Test Script
Chạy script test nhanh để kiểm tra các chức năng chính:
```bash
python quick_test.py
```

**Yêu cầu trước khi chạy:**
1. ✅ Redis đang chạy
2. ✅ Celery worker đang chạy
3. ✅ FastAPI server đang chạy

### Manual Testing
Xem hướng dẫn chi tiết: [TESTING_GUIDE.md](TESTING_GUIDE.md)

### Unit Tests
```bash
pytest tests/test_summarizer.py
```

## 📚 Tài liệu tham khảo
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [CrewAI Documentation](https://docs.crewai.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [Celery Documentation](https://docs.celeryq.dev/)
