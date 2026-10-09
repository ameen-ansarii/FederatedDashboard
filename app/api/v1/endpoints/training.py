import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_user
from app.core.config import settings
from app.models.domain import Job, JobStatus, User
from app.schemas.job import JobResponse
from app.schemas.training import CentralizedTrainingConfig, TrainingRunResponse
from app.services.preprocessing_service import PreprocessingService
from app.services.training_service import TrainingService
from app.tasks.training_tasks import run_centralized_training_task

router = APIRouter()


@router.post("/centralized", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def trigger_centralized_training(
    config: CentralizedTrainingConfig,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Verify preprocessing artifact exists
    PreprocessingService.get_artifact(db, config.artifact_id)

    # Create durable Job record
    job_id = f"job-{uuid.uuid4()}"
    job = Job(
        id=job_id,
        type="CENTRALIZED_TRAINING",
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
            run_centralized_training_task(
                job_id=job_id,
                config_dict=config.model_dump(),
                user_id=current_user.id if current_user else None,
                db_session=db,
            )
        else:
            run_centralized_training_task.apply_async(kwargs=task_kwargs, expires=3600)
    except Exception:
        run_centralized_training_task(
            job_id=job_id,
            config_dict=config.model_dump(),
            user_id=current_user.id if current_user else None,
            db_session=db,
        )

    db.refresh(job)
    return job


@router.get("/runs/{run_id}", response_model=TrainingRunResponse)
def get_training_run(
    run_id: str,
    db: Session = Depends(get_db),
):
    run = TrainingService.get_training_run(db, run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training run with ID {run_id} not found",
        )
    return run
