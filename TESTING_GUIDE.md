# 🧪 Hướng Dẫn Test API - Step by Step

## 📋 Mục Lục
1. [Kiểm Tra Môi Trường](#1-kiểm-tra-môi-trường)
2. [Setup Database](#2-setup-database)
3. [Chạy Services](#3-chạy-services)
4. [Test API - Từ Đơn Giản Đến Phức Tạp](#4-test-api---từ-đơn-giản-đến-phức-tạp)
5. [Kiểm Tra Database](#5-kiểm-tra-database)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Kiểm Tra Môi Trường

### 1.1. Kiểm tra Python & Dependencies
```bash
# Kiểm tra Python version (cần >= 3.10)
python --version

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Cài đặt dependencies (nếu chưa cài)
pip install -r requirements.txt
```

### 1.2. Kiểm tra file `.env`
Đảm bảo file `.env` có đầy đủ:
```env
GOOGLE_API_KEY=your_google_api_key_here
DATABASE_URL=postgresql://user:password@localhost:5432/note_ai
REDIS_URL=redis://localhost:6379/0
TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe
```

### 1.3. Kiểm tra Tesseract OCR
```bash
# Chạy script kiểm tra
python check_tesseract.py

# Hoặc test trực tiếp
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

### 1.4. Kiểm tra FFmpeg (cho Whisper)
```bash
# Test FFmpeg
ffmpeg -version

# Nếu không có, thêm vào PATH hoặc cài đặt
```

---

## 2. Setup Database

### 2.1. Tạo Database PostgreSQL
```sql
-- Kết nối PostgreSQL
psql -U postgres

-- Tạo database
CREATE DATABASE note_ai;

-- Thoát
\q
```

### 2.2. Chạy Migration
```bash
# Nếu database MỚI (chưa có dữ liệu)
python -m app.database.init_db

# Nếu database CŨ (đã có dữ liệu, cần thêm cột mới)
python -m app.database.migrations
```

**Kết quả mong đợi:**
```
✅ Database initialized successfully!
Tables created: users, notes, feedbacks
```

---

## 3. Chạy Services

### 3.1. Terminal 1: Redis (bắt buộc cho async)
```bash
# Windows (nếu cài Redis)
redis-server

# Hoặc dùng Docker
docker run -d -p 6379:6379 redis:latest

# Kiểm tra Redis đang chạy
redis-cli ping
# Kết quả: PONG
```

### 3.2. Terminal 2: Celery Worker (cho async processing)
```bash
# Activate venv trước
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Chạy Celery worker
celery -A app.services.celery_app.celery_app worker --loglevel=info

# Hoặc dùng script có sẵn
# Windows:
run_worker.bat
# Linux/Mac:
./run_worker.sh
```

**Kết quả mong đợi:**
```
[tasks]
  . process_file_async
  . process_text_async

celery@hostname ready.
```

### 3.3. Terminal 3: FastAPI Server
```bash
# Activate venv
.venv\Scripts\activate  # Windows

# Chạy server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Kết quả mong đợi:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### 3.4. Kiểm tra Server
Mở browser: http://localhost:8000/docs

Bạn sẽ thấy Swagger UI với tất cả endpoints.

---

## 4. Test API - Từ Đơn Giản Đến Phức Tạp

### 4.1. Test 1: Health Check (Đơn Giản Nhất)
```bash
# Test root endpoint
curl http://localhost:8000/

# Kết quả mong đợi:
# {"message":"Note Summarizer AI Backend running","llm":"Google Gemini via LangChain"}
```

### 4.2. Test 2: Summarize Text (Sync - Nhanh)
```bash
curl -X POST "http://localhost:8000/api/v1/summarize" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "note=Hôm nay đội marketing họp về chiến lược mới, ngân sách dự kiến 50 triệu đồng. Cần triển khai quảng cáo trên mạng xã hội và tối ưu website."

# Hoặc dùng PowerShell (Windows):
$body = @{
    note = "Hôm nay đội marketing họp về chiến lược mới, ngân sách dự kiến 50 triệu đồng."
} | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/summarize" -Method Post -Body $body -ContentType "application/json"
```

**Kiểm tra kết quả:**
- ✅ Có `summary` (string)
- ✅ Có `summaries` (dict với `one_sentence`, `short_paragraph`, `bullet_points`)
- ✅ Có `questions` (array)
- ✅ Có `mcqs` (dict với `easy`, `medium`, `hard`)
- ✅ Có `raw_text` và `processed_text`

### 4.3. Test 3: Process Text với User ID (Lưu vào DB)
```bash
curl -X POST "http://localhost:8000/api/v1/process" \
  -F "text=Machine Learning là một nhánh của trí tuệ nhân tạo. Nó sử dụng thuật toán để học từ dữ liệu và đưa ra dự đoán." \
  -F "user_id=test_user_001" \
  -F "note_id=note_001"
```

**Kiểm tra:**
- Response có đầy đủ learning assets
- Note được lưu vào database (xem bước 5)

### 4.4. Test 4: Process Image (OCR)
**Chuẩn bị:** Tạo file ảnh có text (ví dụ: chụp màn hình hoặc scan document)

```bash
# Windows PowerShell
$form = @{
    file = Get-Item "path\to\your\image.jpg"
    user_id = "test_user_001"
    note_id = "note_image_001"
}
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/process" -Method Post -Form $form

# Linux/Mac
curl -X POST "http://localhost:8000/api/v1/process" \
  -F "file=@/path/to/your/image.jpg" \
  -F "user_id=test_user_001" \
  -F "note_id=note_image_001"
```

**Kiểm tra:**
- Response có `raw_text` (text từ OCR)
- `processed_text` đã được chuẩn hóa
- Có đầy đủ summaries/questions/mcqs

### 4.5. Test 5: Process PDF
```bash
# Windows PowerShell
$form = @{
    file = Get-Item "path\to\document.pdf"
    user_id = "test_user_001"
    note_id = "note_pdf_001"
}
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/process" -Method Post -Form $form
```

### 4.6. Test 6: Process Audio (Whisper STT)
**Lưu ý:** Lần đầu chạy sẽ tải model Whisper (~1.4GB), mất vài phút.

```bash
# Windows PowerShell
$form = @{
    file = Get-Item "path\to\audio.mp3"
    user_id = "test_user_001"
    note_id = "note_audio_001"
}
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/process" -Method Post -Form $form
```

### 4.7. Test 7: Async Processing (Background Job)
```bash
# Submit job
curl -X POST "http://localhost:8000/api/v1/process/async" \
  -F "text=Đây là một đoạn text dài để test async processing. Nó sẽ được xử lý trong background worker và không block API." \
  -F "user_id=test_user_001" \
  -F "note_id=note_async_001"

# Response:
# {
#   "job_id": "abc-123-def-456",
#   "status": "pending",
#   "message": "Text đang được xử lý..."
# }
```

**Lưu job_id, sau đó:**

```bash
# Check status (thay {job_id} bằng job_id thực tế)
curl "http://localhost:8000/api/v1/jobs/{job_id}/status"

# Kết quả có thể:
# - {"status": "pending", "progress": 0}
# - {"status": "processing", "progress": 45, "stage": "Generating summary..."}
# - {"status": "completed", "progress": 100, "result": {...}}
```

**Khi status = "completed":**
```bash
# Lấy kết quả
curl "http://localhost:8000/api/v1/jobs/{job_id}/result"
```

### 4.8. Test 8: Lấy Notes từ Database
```bash
# Lấy danh sách notes của user
curl "http://localhost:8000/api/v1/users/test_user_001/notes?limit=10"

# Lấy chi tiết một note
curl "http://localhost:8000/api/v1/notes/note_001"
```

### 4.9. Test 9: Search Notes
```bash
curl "http://localhost:8000/api/v1/users/test_user_001/notes/search?q=marketing"
```

---

## 5. Kiểm Tra Database

### 5.1. Kết nối PostgreSQL
```bash
psql -U postgres -d note_ai
```

### 5.2. Kiểm tra Tables
```sql
-- Xem tất cả tables
\dt

-- Kết quả mong đợi:
-- users
-- notes
-- feedbacks
```

### 5.3. Kiểm tra Notes đã lưu
```sql
-- Xem tất cả notes
SELECT id, note_id, file_type, created_at FROM notes;

-- Xem chi tiết một note (thay 'note_001' bằng note_id thực tế)
SELECT 
    note_id,
    file_type,
    LENGTH(raw_text) as raw_text_length,
    LENGTH(processed_text) as processed_text_length,
    summary IS NOT NULL as has_summary,
    summaries IS NOT NULL as has_summaries,
    questions IS NOT NULL as has_questions,
    mcqs IS NOT NULL as has_mcqs,
    review IS NOT NULL as has_review
FROM notes 
WHERE note_id = 'note_001';

-- Xem nội dung summaries (JSON)
SELECT note_id, summaries FROM notes WHERE note_id = 'note_001';

-- Xem questions
SELECT note_id, questions FROM notes WHERE note_id = 'note_001';

-- Xem MCQs
SELECT note_id, mcqs FROM notes WHERE note_id = 'note_001';
```

**Kiểm tra:**
- ✅ `summaries` không NULL và có cấu trúc `{"one_sentence": "...", "short_paragraph": "...", "bullet_points": [...]}`
- ✅ `questions` không NULL và là array
- ✅ `mcqs` không NULL và có keys `easy`, `medium`, `hard`
- ✅ `processed_text` khác `raw_text` (đã được chuẩn hóa)

---

## 6. Troubleshooting

### 6.1. Lỗi: "GOOGLE_API_KEY is required"
**Nguyên nhân:** Chưa set API key trong `.env`
**Giải pháp:**
- Kiểm tra file `.env` có `GOOGLE_API_KEY=...`
- Restart server sau khi sửa `.env`

### 6.2. Lỗi: "Tesseract OCR chưa được cài đặt"
**Nguyên nhân:** Tesseract chưa cài hoặc path sai
**Giải pháp:**
- Cài Tesseract từ: https://github.com/UB-Mannheim/tesseract/wiki
- Đảm bảo tick "Vietnamese" khi cài
- Set `TESSERACT_CMD` trong `.env` đúng path
- Chạy `python check_tesseract.py` để verify

### 6.3. Lỗi: "Could not connect to Redis"
**Nguyên nhân:** Redis chưa chạy
**Giải pháp:**
- Chạy `redis-server` hoặc `docker run -d -p 6379:6379 redis`
- Kiểm tra `REDIS_URL` trong `.env`

### 6.4. Lỗi: "relation 'notes' does not exist"
**Nguyên nhân:** Database chưa được khởi tạo
**Giải pháp:**
- Chạy `python -m app.database.init_db`
- Hoặc `python -m app.database.migrations` nếu DB cũ

### 6.5. Lỗi: "column 'summaries' does not exist"
**Nguyên nhân:** Database cũ chưa có cột mới
**Giải pháp:**
- Chạy `python -m app.database.migrations`

### 6.6. Response thiếu `questions` hoặc `mcqs`
**Nguyên nhân:** LLM không trả về đúng format
**Giải pháp:**
- Kiểm tra log của Celery worker (xem có error không)
- Kiểm tra `GOOGLE_API_KEY` có hợp lệ
- Thử lại với text ngắn hơn

### 6.7. Audio processing chậm
**Nguyên nhân:** Whisper đang tải model lần đầu
**Giải pháp:**
- Đợi lần đầu (model sẽ cache)
- Lần sau sẽ nhanh hơn

### 6.8. Kiểm tra Logs
```bash
# FastAPI logs: Xem terminal chạy uvicorn
# Celery logs: Xem terminal chạy celery worker
# Database logs: Xem PostgreSQL logs
```

---

## 7. Test Checklist

Trước khi deploy production, đảm bảo:

- [ ] ✅ Tất cả endpoints sync hoạt động
- [ ] ✅ Async processing hoạt động (job status + result)
- [ ] ✅ Text processing tạo đủ summaries/questions/mcqs
- [ ] ✅ Image OCR hoạt động (nếu dùng)
- [ ] ✅ PDF/DOCX extraction hoạt động (nếu dùng)
- [ ] ✅ Audio STT hoạt động (nếu dùng)
- [ ] ✅ Database lưu đầy đủ dữ liệu
- [ ] ✅ Search notes hoạt động
- [ ] ✅ Không có lỗi trong logs

---

## 8. Test Nhanh với Swagger UI

1. Mở http://localhost:8000/docs
2. Chọn endpoint muốn test
3. Click "Try it out"
4. Điền parameters
5. Click "Execute"
6. Xem response

**Ưu điểm:** Không cần viết curl, test trực tiếp trên browser.

---

## 9. Script Test Tự Động (Tùy chọn)

Tạo file `test_api.py`:
```python
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

def test_summarize():
    response = requests.post(
        f"{BASE_URL}/summarize",
        data={"note": "Test note để kiểm tra API."}
    )
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "summaries" in data
    assert "questions" in data
    assert "mcqs" in data
    print("✅ Summarize test passed!")

if __name__ == "__main__":
    test_summarize()
```

Chạy: `python test_api.py`

---

**Chúc bạn test thành công! 🎉**
