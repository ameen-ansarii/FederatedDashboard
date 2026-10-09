import json
import traceback
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from app.celery_app import celery_app
from app.core.database import SessionLocal
from app.models.domain import Job, JobStatus, User
from app.services.model_service import ModelService
from app.services.prediction_service import PredictionService
from app.storage.local import storage


@celery_app.task(name="tasks.run_batch_prediction", bind=True)
def run_batch_prediction_task(
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
        # 1. Update job status to RUNNING
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = JobStatus.RUNNING
            job.progress = 20.0
            db.commit()

        model_id = config_dict["model_id"]
        records = config_dict["records"]
        save_to_history = config_dict.get("save_to_history", False)

        model_record = ModelService.get_model(db, model_id)

        # 2. Run batched inference
        summary, predictions = PredictionService.predict_batch(
            db=db,
            model_id=model_id,
            records=records,
            batch_job_id=job_id,
            save_to_history=save_to_history,
            user_id=user_id,
        )

        # 3. Persist batch results JSON in storage
        batch_artifact_rel_path = f"artifacts/predictions/{job_id}/results.json"
        predictions_json = json.dumps(
            {
                "job_id": job_id,
                "status": "SUCCESS",
                "model_id": model_id,
                "model_version": model_record.version,
                "total_samples": summary.total_samples,
                "summary": summary.model_dump(),
                "predictions": [p.model_dump() for p in predictions],
            },
            indent=2,
        ).encode("utf-8")
        storage.save_file(batch_artifact_rel_path, predictions_json)

        # 4. Update job to SUCCESS
        if job:
            job.status = JobStatus.SUCCESS
            job.progress = 100.0
            job.result = {
                "job_id": job_id,
                "model_id": model_id,
                "model_version": model_record.version,
                "total_samples": summary.total_samples,
                "summary": summary.model_dump(),
                "artifact_key": f"batch_{job_id}",
            }
            db.commit()

        return {
            "status": "SUCCESS",
            "job_id": job_id,
            "total_samples": summary.total_samples,
            "summary": summary.model_dump(),
        }

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
