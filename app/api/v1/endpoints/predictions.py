import json
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.config import settings
from app.core.errors import BadRequestException, NotFoundException
from app.models.domain import Job, JobStatus, RiskLevel, User
from app.schemas.job import JobResponse
from app.schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResultResponse,
    PredictionHistoryItem,
    PredictionHistoryResponse,
    SinglePredictionRequest,
    SinglePredictionResponse,
)
from app.services.model_service import ModelService
from app.services.prediction_service import PredictionService
from app.storage.local import storage
from app.tasks.prediction_tasks import run_batch_prediction_task

router = APIRouter()


@router.post("/single", response_model=SinglePredictionResponse, status_code=status.HTTP_200_OK)
def predict_single_record(
    request: SinglePredictionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Executes real-time student dropout prediction on a single record.
    Uses the exact registered model and its lineage-linked preprocessing pipeline.
    """
    return PredictionService.predict_single(
        db=db,
        model_id=request.model_id,
        features=request.features,
        save_to_history=request.save_to_history,
        user=current_user,
    )


@router.post("/batch", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def trigger_batch_prediction(
    request: BatchPredictionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Submits an asynchronous batch prediction job.
    Returns 202 Accepted with a durable Job ID.
    """
    # Verify model exists before queueing
    ModelService.get_model(db, request.model_id)

    job_id = f"job-{uuid.uuid4()}"
    job = Job(
        id=job_id,
        type="BATCH_PREDICTION",
        status=JobStatus.PENDING,
        progress=0.0,
        user_id=current_user.id if current_user else None,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    task_kwargs = {
        "job_id": job_id,
        "config_dict": request.model_dump(),
        "user_id": current_user.id if current_user else None,
    }

    try:
        if settings.ENVIRONMENT == "testing":
            run_batch_prediction_task(
                job_id=job_id,
                config_dict=request.model_dump(),
                user_id=current_user.id if current_user else None,
                db_session=db,
            )
        else:
            run_batch_prediction_task.apply_async(kwargs=task_kwargs, expires=3600)
    except Exception:
        run_batch_prediction_task(
            job_id=job_id,
            config_dict=request.model_dump(),
            user_id=current_user.id if current_user else None,
            db_session=db,
        )

    db.refresh(job)
    return job


@router.get("/batch/{job_id}/results", response_model=BatchPredictionResultResponse, status_code=status.HTTP_200_OK)
def get_batch_prediction_results(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retrieves the complete results of a completed batch prediction job.
    Never exposes internal storage filesystem paths.
    """
    job = db.query(Job).filter(Job.id == job_id, Job.type == "BATCH_PREDICTION").first()
    if not job:
        raise NotFoundException(f"Batch prediction job with ID '{job_id}' not found")

    if job.status != JobStatus.SUCCESS:
        raise BadRequestException(
            f"Batch prediction job '{job_id}' is not in SUCCESS state (current: {job.status.value})"
        )

    # Read the batch predictions artifact from storage
    batch_artifact_rel_path = f"artifacts/predictions/{job_id}/results.json"
    if not storage.file_exists(batch_artifact_rel_path):
        raise NotFoundException(f"Batch prediction results for job '{job_id}' are missing or expired")

    raw_bytes = storage.read_file(batch_artifact_rel_path)
    data = json.loads(raw_bytes.decode("utf-8"))

    return data


@router.get("/history", response_model=PredictionHistoryResponse, status_code=status.HTTP_200_OK)
def get_prediction_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    model_id: Optional[int] = Query(None),
    risk_level: Optional[RiskLevel] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Returns paginated prediction history with optional filtering by model and risk level.
    """
    items, total = PredictionService.list_history(
        db=db,
        skip=skip,
        limit=limit,
        model_id=model_id,
        risk_level=risk_level,
        user_id=current_user.id if current_user else None,
    )

    history_items = []
    for item in items:
        history_items.append(
            PredictionHistoryItem(
                id=item.id,
                model_id=item.model_id,
                model_version=item.model.version if item.model else None,
                probability=item.probability,
                prediction_class=item.prediction_class,
                prediction_label=item.prediction_label,
                risk_level=item.risk_level,
                feature_values=item.feature_values,
                batch_job_id=item.batch_job_id,
                created_at=item.created_at,
            )
        )

    return PredictionHistoryResponse(
        items=history_items,
        total=total,
        skip=skip,
        limit=limit,
    )


@router.get("/history/{prediction_id}", response_model=PredictionHistoryItem, status_code=status.HTTP_200_OK)
def get_single_prediction_history_record(
    prediction_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retrieves a single prediction history record by ID.
    """
    item = PredictionService.get_prediction_by_id(db, prediction_id)
    return PredictionHistoryItem(
        id=item.id,
        model_id=item.model_id,
        model_version=item.model.version if item.model else None,
        probability=item.probability,
        prediction_class=item.prediction_class,
        prediction_label=item.prediction_label,
        risk_level=item.risk_level,
        feature_values=item.feature_values,
        batch_job_id=item.batch_job_id,
        created_at=item.created_at,
    )
