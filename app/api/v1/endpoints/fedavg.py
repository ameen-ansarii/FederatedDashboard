import uuid
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.config import settings
from app.models.domain import Job, JobStatus, User
from app.schemas.federated import FedAvgTrainingConfig
from app.schemas.job import JobResponse
from app.services.preprocessing_service import PreprocessingService
from app.tasks.fedavg_tasks import run_fedavg_training_task

router = APIRouter()


@router.post("/fedavg", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def trigger_fedavg_training(
    config: FedAvgTrainingConfig,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Verify preprocessing artifact exists
    PreprocessingService.get_artifact(db, config.artifact_id)

    # Create durable Job record
    job_id = f"job-{uuid.uuid4()}"
    job = Job(
        id=job_id,
        type="FEDAVG_TRAINING",
        status=JobStatus.PENDING,
        progress=0.0,
        user_id=current_user.id if current_user else None,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    task_kwargs = {
        "job_id": job_id,
        "config_dict": config.model_dump(),
        "user_id": current_user.id if current_user else None,
    }

    try:
        if settings.ENVIRONMENT == "testing":
            run_fedavg_training_task(
                job_id=job_id,
                config_dict=config.model_dump(),
                user_id=current_user.id if current_user else None,
                db_session=db,
            )
        else:
            run_fedavg_training_task.apply_async(kwargs=task_kwargs, expires=3600)
    except Exception:
        run_fedavg_training_task(
            job_id=job_id,
            config_dict=config.model_dump(),
            user_id=current_user.id if current_user else None,
            db_session=db,
        )

    db.refresh(job)
    return job
