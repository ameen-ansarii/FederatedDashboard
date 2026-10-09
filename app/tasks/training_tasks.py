import traceback
from typing import Any, Dict, Optional
from sqlalchemy.orm import Session

from app.celery_app import celery_app
from app.core.database import SessionLocal
from app.models.domain import Job, JobStatus, User
from app.schemas.training import CentralizedTrainingConfig
from app.services.training_service import TrainingService


@celery_app.task(name="tasks.run_centralized_training", bind=True)
def run_centralized_training_task(
    self,
    job_id: str,
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

        user = db.query(User).filter(User.id == user_id).first() if user_id else None
        config = CentralizedTrainingConfig(**config_dict)

        model_record, training_run = TrainingService.run_centralized_training(
            db=db, run_id=job_id, config=config, user=user
        )

        # Update job status to SUCCESS
        if job:
            job.status = JobStatus.SUCCESS
            job.progress = 100.0
            job.result = {
                "model_id": model_record.id,
                "version": model_record.version,
                "metrics": model_record.metrics,
                "checksum": model_record.checksum,
                "training_run_id": training_run.id,
            }
            db.commit()

        return {"status": "SUCCESS", "model_id": model_record.id, "version": model_record.version}

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
