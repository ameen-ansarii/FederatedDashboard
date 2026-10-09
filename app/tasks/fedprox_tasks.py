from typing import Any, Dict, Optional
from sqlalchemy.orm import Session

from app.celery_app import celery_app
from app.core.database import SessionLocal
from app.models.domain import Job, JobStatus, User
from app.schemas.federated import FedProxTrainingConfig
from app.services.fedprox_service import FedProxService


@celery_app.task(name="tasks.run_fedprox_training", bind=True)
def run_fedprox_training_task(
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
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = JobStatus.RUNNING
            job.progress = 10.0
            db.commit()

        user = db.query(User).filter(User.id == user_id).first() if user_id else None
        config = FedProxTrainingConfig(**config_dict)

        model_record, training_run = FedProxService.run_fedprox_training(
            db=db, run_id=job_id, config=config, user=user
        )

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
