import traceback
from typing import Any, Dict, Optional
from sqlalchemy.orm import Session

from app.celery_app import celery_app
from app.core.database import SessionLocal
from app.models.domain import Experiment, Job, JobStatus, User
from app.schemas.experiment import BenchmarkConfig
from app.services.experiment_service import ExperimentService


@celery_app.task(name="tasks.run_experiment_benchmark", bind=True)
def run_experiment_benchmark_task(
    self,
    job_id: str,
    experiment_id: str,
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
        config = BenchmarkConfig(**config_dict)

        experiment = ExperimentService.run_benchmark(
            db=db,
            experiment_id=experiment_id,
            config=config,
            user=user,
        )

        if job:
            job.status = JobStatus.SUCCESS
            job.progress = 100.0
            job.result = {
                "experiment_id": experiment.id,
                "name": experiment.name,
                "status": experiment.status.value,
                "summary": experiment.results.get("summary", {}) if experiment.results else {},
            }
            db.commit()

        return {
            "status": "SUCCESS",
            "experiment_id": experiment.id,
            "summary": experiment.results.get("summary", {}) if experiment.results else {},
        }

    except Exception as e:
        db.rollback()
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = JobStatus.FAILED
            job.error = str(e)
            db.commit()

        exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if exp:
            exp.status = JobStatus.FAILED
            db.commit()

        raise e
    finally:
        if own_session:
            db.close()
