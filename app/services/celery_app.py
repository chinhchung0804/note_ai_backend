"""
Celery App Configuration
Background task queue for async processing
"""
import os
import platform

from celery import Celery

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, skip
    pass

REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Celery configuration
celery_app = Celery(
    'note_ai_worker',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=['app.services.tasks'],
)

# Celery settings
# Detect Windows and use solo pool (Windows doesn't support prefork)
is_windows = platform.system() == 'Windows'
worker_pool = 'solo' if is_windows else 'prefork'

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,  
    task_time_limit=300,  
    task_soft_time_limit=270, 
    result_expires=3600,  
    worker_pool=worker_pool,  
)

# Auto-discover tasks inside services package
celery_app.autodiscover_tasks(['app.services'])

__all__ = ('celery_app',)