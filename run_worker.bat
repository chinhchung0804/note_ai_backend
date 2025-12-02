@echo off
REM Script để chạy Celery worker trên Windows

REM Activate virtualenv nếu có
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

REM Start Celery worker (Windows needs solo pool to avoid WinError 5)
celery -A app.services.celery_app worker --loglevel=info --pool=solo

