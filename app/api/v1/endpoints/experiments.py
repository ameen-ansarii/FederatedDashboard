import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.config import settings
from app.core.errors import BadRequestException, NotFoundException
from app.models.domain import Experiment, Job, JobStatus, User
from app.schemas.experiment import (
    BenchmarkConfig,
    ConvergenceTrajectoryResponse,
    ExperimentListItem,
    ExperimentListResponse,
    ExperimentResponse,
    ModelComparisonRequest,
    ModelEvaluationSummary,
    MuSensitivityResponse,
)
from app.schemas.job import JobResponse
from app.services.experiment_service import ExperimentService
from app.services.preprocessing_service import PreprocessingService
from app.tasks.experiment_tasks import run_experiment_benchmark_task

router = APIRouter()


@router.post("/benchmark", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def trigger_experiment_benchmark(
    config: BenchmarkConfig,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Submits an end-to-end controlled benchmark comparing Centralized, FedAvg (IID & Non-IID),
    and FedProx across multiple μ values under the exact same preprocessing artifact and test split.
    """
    # 1. Validate artifact exists
    PreprocessingService.get_artifact(db, config.artifact_id)

    # 2. Create durable Job and Experiment records
    job_id = f"job-{uuid.uuid4()}"
    experiment_id = f"exp-{uuid.uuid4()}"

    job = Job(
        id=job_id,
        type="EXPERIMENT_BENCHMARK",
        status=JobStatus.PENDING,
        progress=0.0,
        user_id=current_user.id if current_user else None,
    )
    db.add(job)

    experiment = Experiment(
        id=experiment_id,
        name=config.name or "FL vs Centralized Benchmark",
        status=JobStatus.PENDING,
        artifact_id=config.artifact_id,
        config=config.model_dump(),
        user_id=current_user.id if current_user else None,
    )
    db.add(experiment)
    db.commit()
    db.refresh(job)

    task_kwargs = {
        "job_id": job_id,
        "experiment_id": experiment_id,
        "config_dict": config.model_dump(),
        "user_id": current_user.id if current_user else None,
    }

    try:
        if settings.ENVIRONMENT == "testing":
            run_experiment_benchmark_task(
                job_id=job_id,
                experiment_id=experiment_id,
                config_dict=config.model_dump(),
                user_id=current_user.id if current_user else None,
                db_session=db,
            )
        else:
            run_experiment_benchmark_task.apply_async(kwargs=task_kwargs, expires=7200)
    except Exception:
        run_experiment_benchmark_task(
            job_id=job_id,
            experiment_id=experiment_id,
            config_dict=config.model_dump(),
            user_id=current_user.id if current_user else None,
            db_session=db,
        )

    db.refresh(job)
    return job


@router.post("/compare", response_model=List[ModelEvaluationSummary], status_code=status.HTTP_200_OK)
def compare_registered_models(
    request: ModelComparisonRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Evaluates and compares arbitrary registered models side-by-side on the same held-out test split.
    """
    return ExperimentService.compare_models(db=db, request=request, user=current_user)


@router.get("", response_model=ExperimentListResponse, status_code=status.HTTP_200_OK)
def list_experiments(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Lists historical benchmark experiments.
    """
    items, total = ExperimentService.list_experiments(db=db, skip=skip, limit=limit, user=current_user)
    list_items = []
    for exp in items:
        results = exp.results or {}
        models = results.get("models", [])
        list_items.append(
            ExperimentListItem(
                id=exp.id,
                name=exp.name,
                status=exp.status.value,
                artifact_id=exp.artifact_id,
                created_at=exp.created_at,
                model_count=len(models),
            )
        )
    return ExperimentListResponse(items=list_items, total=total, skip=skip, limit=limit)


@router.get("/{experiment_id}", response_model=ExperimentResponse, status_code=status.HTTP_200_OK)
def get_experiment_details(
    experiment_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retrieves full benchmark experiment results, models summary, and metrics deltas.
    """
    exp = ExperimentService.get_experiment_by_id(db=db, experiment_id=experiment_id, user=current_user)
    results = exp.results or {}
    models_raw = results.get("models", [])

    model_objs = []
    for m in models_raw:
        model_objs.append(ModelEvaluationSummary(**m))

    return ExperimentResponse(
        id=exp.id,
        name=exp.name,
        status=exp.status.value,
        artifact_id=exp.artifact_id,
        config=exp.config or {},
        summary=results.get("summary", {}),
        models=model_objs,
        client_distributions=results.get("client_distributions"),
        created_at=exp.created_at,
    )


@router.get("/{experiment_id}/convergence", response_model=ConvergenceTrajectoryResponse, status_code=status.HTTP_200_OK)
def get_experiment_convergence_trajectories(
    experiment_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retrieves round-by-round and epoch-by-epoch loss & validation accuracy curves for cross-method charting.
    """
    exp = ExperimentService.get_experiment_by_id(db=db, experiment_id=experiment_id, user=current_user)
    results = exp.results or {}
    trajectories = results.get("trajectories", {})
    return ConvergenceTrajectoryResponse(experiment_id=exp.id, trajectories=trajectories)


@router.get("/{experiment_id}/mu-sensitivity", response_model=MuSensitivityResponse, status_code=status.HTTP_200_OK)
def get_experiment_mu_sensitivity_analysis(
    experiment_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retrieves FedProx μ sensitivity evaluation comparing accuracy, F1, ROC-AUC, and loss across tested μ values.
    """
    exp = ExperimentService.get_experiment_by_id(db=db, experiment_id=experiment_id, user=current_user)
    results = exp.results or {}
    mu_analysis = results.get("mu_analysis", [])
    return MuSensitivityResponse(experiment_id=exp.id, mu_analysis=mu_analysis)
