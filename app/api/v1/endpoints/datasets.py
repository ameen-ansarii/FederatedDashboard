import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_user
from app.models.domain import Dataset, Job, JobStatus, PreprocessingArtifact, User
from app.schemas.dataset import DatasetDetailResponse, DatasetResponse
from app.schemas.job import JobResponse
from app.schemas.preprocessing import PreprocessingArtifactResponse, PreprocessingConfig
from app.services.dataset_service import DatasetService
from app.services.preprocessing_service import PreprocessingService
from app.tasks.preprocessing_tasks import run_preprocessing_task

router = APIRouter()


@router.post("/upload", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
def upload_dataset(
    name: str = Form(...),
    target_column: str = Form("Target"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    content = file.file.read()
    dataset = DatasetService.save_and_create_dataset(
        db=db,
        name=name,
        filename=file.filename or "dataset.csv",
        content=content,
        user=current_user,
        target_column=target_column,
    )
    return dataset


@router.get("", response_model=List[DatasetResponse])
def list_datasets(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    datasets = db.query(Dataset).offset(skip).limit(limit).all()
    return datasets


@router.get("/{dataset_id}", response_model=DatasetDetailResponse)
def get_dataset_detail(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    detail = DatasetService.get_dataset_detail(db, dataset_id)
    return detail


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    DatasetService.delete_dataset(db, dataset_id)
    return None


@router.post("/{dataset_id}/preprocess", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def trigger_preprocessing(
    dataset_id: int,
    config: PreprocessingConfig,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    # Verify dataset exists
    DatasetService.get_dataset_by_id(db, dataset_id)

    # Create durable Job record
    job_id = f"job-{uuid.uuid4()}"
    job = Job(
        id=job_id,
        type="PREPROCESSING",
        status=JobStatus.PENDING,
        progress=0.0,
        user_id=current_user.id if current_user else None,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Dispatch Celery task
    task_kwargs = {
        "job_id": job_id,
        "dataset_id": dataset_id,
        "config_dict": config.model_dump(),
        "user_id": current_user.id if current_user else None,
    }
    
    # Execute task (sync execution passing active db session for test/dev, or async delay for celery)
    try:
        if settings.ENVIRONMENT == "testing":
            run_preprocessing_task(job_id=job_id, dataset_id=dataset_id, config_dict=config.model_dump(), user_id=current_user.id if current_user else None, db_session=db)
        else:
            run_preprocessing_task.apply_async(kwargs=task_kwargs, expires=3600)
    except Exception:
        run_preprocessing_task(job_id=job_id, dataset_id=dataset_id, config_dict=config.model_dump(), user_id=current_user.id if current_user else None, db_session=db)

    db.refresh(job)
    return job


@router.get("/{dataset_id}/artifacts", response_model=List[PreprocessingArtifactResponse])
def list_preprocessing_artifacts(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    artifacts = (
        db.query(PreprocessingArtifact)
        .filter(PreprocessingArtifact.dataset_id == dataset_id)
        .order_by(PreprocessingArtifact.created_at.desc())
        .all()
    )
    return artifacts


@router.get("/artifacts/{artifact_id}/schema", response_model=PreprocessingArtifactResponse)
def get_artifact_schema(
    artifact_id: int,
    db: Session = Depends(get_db),
):
    artifact = PreprocessingService.get_artifact(db, artifact_id)
    return artifact
