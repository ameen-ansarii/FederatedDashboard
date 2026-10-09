from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "dropout_fl_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max for training / XAI jobs
)


@celery_app.task(name="ping")
def ping_worker() -> str:
    return "pong"
