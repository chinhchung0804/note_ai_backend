#!/bin/bash
# Script để chạy Celery worker locally

# Activate virtualenv nếu có
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Start Celery worker
celery -A app.services.celery_app worker --loglevel=info --concurrency=2

