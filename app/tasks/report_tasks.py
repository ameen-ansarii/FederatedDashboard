from typing import Any, Dict, Optional
from sqlalchemy.orm import Session

from app.celery_app import celery_app
from app.core.database import SessionLocal
from app.models.domain import Job, JobStatus, PublicationReport, User
from app.schemas.report import ReportGenerationRequest
from app.services.report_service import ReportService


@celery_app.task(name="tasks.generate_publication_report", bind=True)
def generate_publication_report_task(
    self,
    job_id: str,
    report_id: str,
    request_dict: Dict[str, Any],
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
            job.progress = 20.0
            db.commit()

        user = db.query(User).filter(User.id == user_id).first() if user_id else None
        request = ReportGenerationRequest(**request_dict)

        report_resp = ReportService.generate_report(
            db=db,
            report_id=report_id,
            request=request,
            user=user,
        )

        if job:
            job.status = JobStatus.SUCCESS
            job.progress = 100.0
            job.result = {
                "report_id": report_resp.id,
                "name": report_resp.name,
                "status": report_resp.status.value,
                "generated_artifact_path": report_resp.generated_artifact_path,
                "report_metadata": report_resp.report_metadata,
            }
            db.commit()

        return {
            "status": "SUCCESS",
            "report_id": report_resp.id,
            "name": report_resp.name,
        }

    except Exception as e:
        db.rollback()
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = JobStatus.FAILED
            job.error = str(e)
            db.commit()

        rep = db.query(PublicationReport).filter(PublicationReport.id == report_id).first()
        if rep:
            rep.status = JobStatus.FAILED
            db.commit()

        raise e
    finally:
        if own_session:
            db.close()
