from fastapi import APIRouter, Depends, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text
import redis
from app.core.config import settings
from app.core.database import get_db
from app.storage.local import storage
from app.celery_app import celery_app

router = APIRouter()


class ComponentStatus(BaseModel):
    status: str
    message: str = "Healthy"


class HealthCheckResponse(BaseModel):
    status: str
    components: dict[str, ComponentStatus]


@router.get("/health", response_model=HealthCheckResponse, status_code=status.HTTP_200_OK)
def check_health(db: Session = Depends(get_db)):
    components: dict[str, ComponentStatus] = {}
    overall_ok = True

    # 1. API status
    components["api"] = ComponentStatus(status="ok", message="FastAPI application running")

    # 2. Database check (PostgreSQL)
    try:
        db.execute(text("SELECT 1"))
        components["database"] = ComponentStatus(status="ok", message="PostgreSQL operational")
    except Exception as e:
        overall_ok = False
        components["database"] = ComponentStatus(status="error", message=f"Database connection failed: {str(e)}")

    # 3. Redis check
    try:
        r = redis.Redis.from_url(settings.REDIS_URL, socket_timeout=0.5)
        r.ping()
        components["redis"] = ComponentStatus(status="ok", message="Redis operational")
    except Exception as e:
        overall_ok = False
        components["redis"] = ComponentStatus(status="error", message=f"Redis connection failed: {str(e)}")

    # 4. Celery Worker check
    try:
        inspect = celery_app.control.inspect(timeout=0.5)
        active_workers = inspect.ping()
        if active_workers:
            components["worker"] = ComponentStatus(status="ok", message=f"Celery workers active: {list(active_workers.keys())}")
        else:
            components["worker"] = ComponentStatus(status="ok", message="Celery broker connected (no active ping response)")
    except Exception as e:
        components["worker"] = ComponentStatus(status="warning", message=f"Celery worker inspect warning: {str(e)}")

    # 5. Storage check
    try:
        test_file = "_health_check_test.tmp"
        storage.save_file(test_file, b"health_check")
        if storage.file_exists(test_file):
            storage.delete_file(test_file)
            components["storage"] = ComponentStatus(status="ok", message="Storage directory read/write operational")
        else:
            overall_ok = False
            components["storage"] = ComponentStatus(status="error", message="Storage write test failed")
    except Exception as e:
        overall_ok = False
        components["storage"] = ComponentStatus(status="error", message=f"Storage check failed: {str(e)}")

    overall_status = "ok" if overall_ok else "unhealthy"
    return HealthCheckResponse(status=overall_status, components=components)
