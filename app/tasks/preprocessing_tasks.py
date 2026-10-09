import traceback
from typing import Any, Dict, Optional
from sqlalchemy.orm import Session

from app.celery_app import celery_app
from app.core.config import settings
from app.core.database import SessionLocal
from app.models.domain import Job, JobStatus
from app.schemas.preprocessing import PreprocessingConfig
from app.services.preprocessing_service import PreprocessingService


@celery_app.task(name="tasks.run_preprocessing", bind=True)
def run_preprocessing_task(
    self,
    job_id: str,
    dataset_id: int,
    config_dict: Dict[str, Any],
    user_id: Optional[int] = None,
    db_session: Optional[Session] = None,
):
    own_session = False
    if db_session is None:
        db: Session = SessionLocal()
        own_session = True
    else:
        db = db_session

    try:
        # Update job status to RUNNING
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = JobStatus.RUNNING
            job.progress = 10.0
            db.commit()

        config = PreprocessingConfig(**config_dict)
        artifact = PreprocessingService.preprocess_dataset(
            db=db, dataset_id=dataset_id, config=config, user=None
        )

        # Update job status to SUCCESS
        if job:
            job.status = JobStatus.SUCCESS
            job.progress = 100.0
            job.result = {
                "artifact_id": artifact.id,
                "version": artifact.version,
                "feature_schema": artifact.feature_schema,
                "split_summary": artifact.split_summary,
            }
            db.commit()
        return {"status": "SUCCESS", "artifact_id": artifact.id}
    except Exception as e:
        db.rollback()
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = JobStatus.FAILED
            job.error = str(e)
            db.commit()
        raise e
    finally:
        if own_session:
            db.close()
