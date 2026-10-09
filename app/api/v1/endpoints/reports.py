import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.config import settings
from app.core.errors import ForbiddenException, NotFoundException
from app.models.domain import Job, JobStatus, PublicationReport, User, UserRole
from app.schemas.job import JobResponse
from app.schemas.report import (
    PublicationArtifactResponse,
    ReportGenerationRequest,
    ReportListResponse,
    ReportResponse,
    ReportSummaryResponse,
    ReproducibilityManifest,
)
from app.services.report_service import ReportService
from app.storage.local import storage
from app.tasks.report_tasks import generate_publication_report_task

router = APIRouter()


@router.post("", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def generate_publication_report(
    request: ReportGenerationRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """
    Submits a background task to generate a publication-quality research report,
    XAI synthesis, publication figures, CSV metrics, and reproducibility manifests.
    """
    # 1. Authorization check
    if current_user and current_user.role not in [UserRole.ADMIN, UserRole.RESEARCHER]:
        raise ForbiddenException("Only ADMIN or RESEARCHER roles can generate publication reports")

    # 2. Lineage Validation before creating job
    ReportService.validate_report_lineage(db, request.experiment_id)

    # 3. Create durable Job and PublicationReport records
    job_id = f"job-{uuid.uuid4()}"
    report_id = f"report-{uuid.uuid4()}"

    job = Job(
        id=job_id,
        type="PUBLICATION_REPORT_GENERATION",
        status=JobStatus.PENDING,
        progress=0.0,
        user_id=current_user.id if current_user else None,
    )
    db.add(job)

    report_record = PublicationReport(
        id=report_id,
        experiment_id=request.experiment_id,
        name=request.name or "Publication Research Report",
        status=JobStatus.PENDING,
        configuration=request.model_dump(),
        user_id=current_user.id if current_user else None,
    )
    db.add(report_record)
    db.commit()
    db.refresh(job)

    task_kwargs = {
        "job_id": job_id,
        "report_id": report_id,
        "request_dict": request.model_dump(),
        "user_id": current_user.id if current_user else None,
    }

    try:
        if settings.ENVIRONMENT == "testing":
            generate_publication_report_task(
                job_id=job_id,
                report_id=report_id,
                request_dict=request.model_dump(),
                user_id=current_user.id if current_user else None,
                db_session=db,
            )
        else:
            generate_publication_report_task.apply_async(kwargs=task_kwargs, expires=3600)
    except Exception:
        generate_publication_report_task(
            job_id=job_id,
            report_id=report_id,
            request_dict=request.model_dump(),
            user_id=current_user.id if current_user else None,
            db_session=db,
        )

    db.refresh(job)
    return job


@router.get("", response_model=ReportListResponse)
def list_reports(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    experiment_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """
    Lists all generated publication reports.
    """
    query = db.query(PublicationReport)
    if experiment_id:
        query = query.filter(PublicationReport.experiment_id == experiment_id)

    total = query.count()
    items = query.order_by(PublicationReport.created_at.desc()).offset(skip).limit(limit).all()

    return ReportListResponse(
        items=[ReportResponse.model_validate(item) for item in items],
        total=total,
    )


@router.get("/{report_id}", response_model=ReportResponse)
def get_report_metadata(
    report_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """
    Retrieves status and metadata for a specific publication report.
    """
    return ReportResponse.model_validate(ReportService.get_report(db, report_id))


@router.get("/{report_id}/summary", response_model=ReportSummaryResponse)
def get_report_summary(
    report_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """
    Retrieves machine-readable summary metrics, model rankings, and XAI synthesis.
    """
    return ReportService.get_report_summary(db, report_id)


@router.get("/{report_id}/artifacts", response_model=PublicationArtifactResponse)
def list_report_artifacts(
    report_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """
    Lists all generated publication files and figures with their SHA-256 checksums.
    """
    return ReportService.get_report_artifacts(db, report_id)


@router.get("/{report_id}/manifest", response_model=ReproducibilityManifest)
def get_reproducibility_manifest(
    report_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """
    Retrieves the complete reproducibility manifest tracking model checksums, seeds, and lineages.
    """
    return ReportService.get_report_manifest(db, report_id)


@router.get("/{report_id}/artifacts/{artifact_filename}")
def download_report_artifact(
    report_id: str,
    artifact_filename: str,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """
    Streams a generated publication artifact file (report.md, report.json, CSV, or PNG figure).
    """
    report = ReportService.get_report(db, report_id)
    file_key = f"{report.generated_artifact_path}/{artifact_filename}"

    try:
        content_bytes = storage.read_file(file_key)
    except Exception:
        raise NotFoundException(f"Artifact {artifact_filename} not found for report {report_id}")

    media_type = "application/octet-stream"
    if artifact_filename.endswith(".md"):
        media_type = "text/markdown"
    elif artifact_filename.endswith(".json"):
        media_type = "application/json"
    elif artifact_filename.endswith(".csv"):
        media_type = "text/csv"
    elif artifact_filename.endswith(".png"):
        media_type = "image/png"

    return Response(
        content=content_bytes,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{artifact_filename}"'},
    )
