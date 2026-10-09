import io
import json
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.domain import Experiment, JobStatus, ModelRegistry, PreprocessingArtifact
from app.schemas.report import ReportGenerationRequest
from app.services.report_service import ReportService
from tests.test_reporting import create_reporting_mock_data


def test_lineage_preservation_in_manifest_and_report(db: Session, client: TestClient):
    """
    TEST 9 & 10: Lineage & Reproducibility Manifest
    Verify the report and reproducibility manifest retain complete lineage:
    experiment_id, preprocessing_artifact_version, seeds, derived seeds, model checksums.
    """
    exp_id, artifact_id, model_id = create_reporting_mock_data(db, client)
    report_id = "report-test-manifest-001"

    req = ReportGenerationRequest(
        experiment_id=exp_id,
        include_xai=False,
        include_reproducibility_manifest=True,
    )
    ReportService.generate_report(db=db, report_id=report_id, request=req)

    manifest = ReportService.get_report_manifest(db, report_id)

    assert manifest.experiment_id == exp_id
    assert manifest.preprocessing_artifact_id == artifact_id
    assert manifest.seeds == [42, 123, 456, 789, 1024]
    assert len(manifest.derived_seeds_by_seed) == 5
    assert len(manifest.model_checksums_by_seed) == 5

    # Check that each seed has all model variants mapped
    for s in [42, 123, 456, 789, 1024]:
        s_key = str(s)
        assert s_key in manifest.derived_seeds_by_seed
        assert s_key in manifest.model_checksums_by_seed
        assert len(manifest.derived_seeds_by_seed[s_key]) >= 6
        assert len(manifest.model_checksums_by_seed[s_key]) >= 6

    # Verify architecture metadata
    assert manifest.architecture["model_type"] == "DropoutMLP"
    assert manifest.hyperparameters["optimizer"] == "adam"
    assert manifest.hyperparameters["learning_rate"] == 0.01
    assert manifest.system_runtime["seed_isolation_engine"] == "app.core.reproducibility.py"
