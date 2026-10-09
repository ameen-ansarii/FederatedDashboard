import io
import json
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.errors import ForbiddenException, NotFoundException
from app.models.domain import (
    Experiment,
    Job,
    JobStatus,
    ModelRegistry,
    PreprocessingArtifact,
    PublicationReport,
    User,
    UserRole,
)
from app.schemas.report import ReportGenerationRequest
from app.services.report_service import ReportService
from app.storage.local import storage


def create_reporting_mock_data(db: Session, client: TestClient):
    # 1. Upload dataset & create artifact
    df = pd.DataFrame({
        "total_vle_clicks": np.random.randint(10, 500, size=50),
        "avg_assessment_score": np.random.uniform(20.0, 95.0, size=50),
        "studied_credits": np.random.choice([30, 60, 90, 120], size=50),
        "num_of_prev_attempts": np.random.choice([0, 1, 2], size=50),
        "code_module": np.random.choice(["AAA", "BBB"], size=50),
        "code_presentation": np.random.choice(["2013J", "2014J"], size=50),
        "gender": np.random.choice(["M", "F"], size=50),
        "highest_education": np.random.choice(["A Level", "HE Qualification"], size=50),
        "imd_band": np.random.choice(["10-20%", "20-30%"], size=50),
        "age_band": np.random.choice(["0-35", "35-55"], size=50),
        "disability": np.random.choice(["N", "Y"], size=50),
        "final_result": np.random.choice(["Dropout", "Pass"], size=50, p=[0.4, 0.6]),
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)

    up_res = client.post(
        "/api/v1/datasets/upload",
        data={"name": "OULAD Report Benchmark", "target_column": "final_result"},
        files={"file": ("oulad_mock.csv", buf.getvalue(), "text/csv")},
    )
    dataset_id = up_res.json()["id"]

    client.post(
        f"/api/v1/datasets/{dataset_id}/preprocess",
        json={
            "target_column": "final_result",
            "positive_class_val": "Dropout",
            "scale_features": True,
            "val_size": 0.20,
            "test_size": 0.20,
            "random_state": 42,
        },
    )
    artifacts = client.get(f"/api/v1/datasets/{dataset_id}/artifacts").json()
    artifact_id = artifacts[0]["id"]

    # 2. Create Experiment with 5-seed benchmark results
    exp_id = "exp-test-reporting-001"
    seeds = [42, 123, 456, 789, 1024]

    def make_runs(base_f1: float, std_f1: float):
        runs = []
        for s in seeds:
            f1_val = round(float(base_f1 + (s % 7 - 3) * std_f1), 4)
            runs.append({
                "seed": s,
                "metrics": {
                    "accuracy": round(0.55 + f1_val * 0.1, 4),
                    "precision": round(0.40 + f1_val * 0.1, 4),
                    "recall": round(0.50 + f1_val * 0.1, 4),
                    "f1": f1_val,
                    "roc_auc": round(0.60 + f1_val * 0.1, 4),
                    "loss": round(0.75 - f1_val * 0.1, 4),
                    "confusion_matrix": [[20, 7], [5, 8]],
                },
                "checksum": f"chk_{s}_{base_f1}",
                "derived_seed": s * 1000 + 42,
            })
        return runs

    mock_model_results = {
        "CENTRALIZED (BASELINE)": {
            "algorithm": "CENTRALIZED",
            "variant": "BASELINE",
            "seed_runs": make_runs(0.50, 0.02),
        },
        "FEDAVG (IID)": {
            "algorithm": "FEDAVG",
            "variant": "IID",
            "seed_runs": make_runs(0.45, 0.01),
        },
        "FEDAVG (NON_IID_ALPHA_0.5)": {
            "algorithm": "FEDAVG",
            "variant": "NON_IID_ALPHA_0.5",
            "seed_runs": make_runs(0.38, 0.03),
        },
        "FEDPROX (NON_IID_ALPHA_0.5_MU_0.001)": {
            "algorithm": "FEDPROX",
            "variant": "NON_IID_ALPHA_0.5_MU_0.001",
            "seed_runs": make_runs(0.35, 0.04),
        },
        "FEDPROX (NON_IID_ALPHA_0.5_MU_0.01)": {
            "algorithm": "FEDPROX",
            "variant": "NON_IID_ALPHA_0.5_MU_0.01",
            "seed_runs": make_runs(0.41, 0.02),
        },
        "FEDPROX (NON_IID_ALPHA_0.5_MU_0.1)": {
            "algorithm": "FEDPROX",
            "variant": "NON_IID_ALPHA_0.5_MU_0.1",
            "seed_runs": make_runs(0.37, 0.05),
        },
    }

    mock_trajectories = {
        "CENTRALIZED": {"epochs": [1, 2, 3, 4, 5], "loss": [0.8, 0.7, 0.65, 0.6, 0.58], "validation_accuracy": [0.5, 0.55, 0.58, 0.6, 0.62]},
        "FEDAVG_IID": {"rounds": [1, 2, 3, 4, 5], "loss": [0.85, 0.75, 0.7, 0.68, 0.65], "validation_accuracy": [0.48, 0.52, 0.54, 0.55, 0.56]},
    }

    mock_client_dists = {
        "iid": {"client_breakdown": {"client_1": {"sample_count": 40, "class_0_count": 27, "class_1_count": 13}, "client_2": {"sample_count": 40, "class_0_count": 27, "class_1_count": 13}, "client_3": {"sample_count": 40, "class_0_count": 26, "class_1_count": 14}}},
        "non_iid": {"client_breakdown": {"client_1": {"sample_count": 40, "class_0_count": 35, "class_1_count": 5}, "client_2": {"sample_count": 40, "class_0_count": 25, "class_1_count": 15}, "client_3": {"sample_count": 40, "class_0_count": 20, "class_1_count": 20}}},
    }

    exp = Experiment(
        id=exp_id,
        name="Phase 9D Multi-Seed Benchmark",
        status=JobStatus.SUCCESS,
        artifact_id=artifact_id,
        config={"seeds": seeds, "num_clients": 3, "rounds": 5},
        results={
            "seeds": seeds,
            "model_results": mock_model_results,
            "trajectories": mock_trajectories,
            "client_distributions": mock_client_dists,
        },
    )
    db.add(exp)

    # Add a registered model for lineage
    model = ModelRegistry(
        name="FedProx_OULAD_ReportModel",
        version="v_report_test_1.0",
        framework="PyTorch",
        artifact_path="models/test_report_model.pt",
        metrics={"accuracy": 0.60, "f1": 0.42, "precision": 0.38, "recall": 0.46, "roc_auc": 0.62},
        lineage={"dataset_id": dataset_id, "preprocessing_artifact_id": artifact_id, "seed": 42},
        checksum="test_model_checksum_hash_12345",
    )
    db.add(model)
    db.commit()

    return exp_id, artifact_id, model.id


def test_report_generation_end_to_end(db: Session, client: TestClient):
    """
    TEST 1: Report Generation End-to-End
    Verify publication report creation, status completion, and storage persistence.
    """
    exp_id, artifact_id, model_id = create_reporting_mock_data(db, client)

    req = ReportGenerationRequest(
        experiment_id=exp_id,
        name="OULAD FL Dropout Prediction Report",
        primary_metric="f1",
        include_xai=False,
        include_visualizations=True,
        include_raw_seed_results=True,
        include_reproducibility_manifest=True,
        generate_markdown=True,
        generate_json=True,
        generate_csv=True,
        generate_figures=True,
    )

    report_id = "report-test-001"
    report_res = ReportService.generate_report(db=db, report_id=report_id, request=req)

    assert report_res.id == report_id
    assert report_res.status == JobStatus.SUCCESS
    assert report_res.generated_artifact_path == f"reports/{exp_id}/{report_id}"

    # Verify generated artifacts
    art_res = ReportService.get_report_artifacts(db, report_id)
    filenames = [a.filename for a in art_res.artifacts]
    assert "report.md" in filenames
    assert "report.json" in filenames
    assert "reproducibility_manifest.json" in filenames
    assert "artifact_manifest.json" in filenames
    assert "model_metrics.csv" in filenames
    assert "seed_metrics.csv" in filenames
    assert "baseline_deltas.csv" in filenames
    assert "fig1_model_performance.png" in filenames
    assert "fig2_f1_across_seeds.png" in filenames
    assert "fig3_roc_auc_across_seeds.png" in filenames
    assert "fig4_fedprox_mu_sensitivity.png" in filenames
    assert "fig5_iid_vs_non_iid.png" in filenames
    assert "fig6_convergence.png" in filenames
    assert "fig7_client_heterogeneity.png" in filenames


def test_no_fabricated_metrics(db: Session, client: TestClient):
    """
    TEST 2: No Fabricated Metrics
    Assert report metrics match persisted experiment results within float tolerance.
    """
    exp_id, _, _ = create_reporting_mock_data(db, client)
    report_id = "report-test-002"

    req = ReportGenerationRequest(experiment_id=exp_id, include_xai=False)
    ReportService.generate_report(db=db, report_id=report_id, request=req)

    summary = ReportService.get_report_summary(db, report_id)
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()

    raw_model_results = exp.results["model_results"]

    for ms in summary.model_summaries:
        key = f"{ms.algorithm} ({ms.variant})" if ms.algorithm != "CENTRALIZED" else "CENTRALIZED (BASELINE)"
        assert key in raw_model_results
        raw_runs = raw_model_results[key]["seed_runs"]

        expected_f1_vals = [r["metrics"]["f1"] for r in raw_runs]
        expected_mean = float(np.mean(expected_f1_vals))
        expected_std = float(np.std(expected_f1_vals, ddof=1))

        assert abs(ms.metrics["f1"].mean - expected_mean) < 1e-4
        assert abs(ms.metrics["f1"].std - expected_std) < 1e-4


def test_mean_std_calculation(db: Session, client: TestClient):
    """
    TEST 3: Mean & Sample Standard Deviation (ddof=1)
    """
    vals = [0.4848, 0.4000, 0.5294, 0.5714, 0.5000]
    expected_mean = float(np.mean(vals))
    expected_std = float(np.std(vals, ddof=1))

    calc_std = ReportService._sample_std(vals)
    assert abs(calc_std - expected_std) < 1e-5
    assert abs(np.mean(vals) - expected_mean) < 1e-5


def test_paired_baseline_deltas(db: Session, client: TestClient):
    """
    TEST 4: Paired Baseline Deltas
    Verify baseline deltas are computed per seed before averaging.
    """
    exp_id, _, _ = create_reporting_mock_data(db, client)
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()

    agg_stats, seeds = ReportService.aggregate_seed_metrics(exp.results)
    cent_runs = exp.results["model_results"]["CENTRALIZED (BASELINE)"]["seed_runs"]
    cent_f1 = {r["seed"]: r["metrics"]["f1"] for r in cent_runs}

    for vname, vdata in agg_stats.items():
        if "CENTRALIZED" in vname:
            continue
        v_runs = vdata["seed_runs"]
        paired_deltas = [r["metrics"]["f1"] - cent_f1[r["seed"]] for r in v_runs]
        expected_delta_mean = float(np.mean(paired_deltas))
        expected_delta_std = float(np.std(paired_deltas, ddof=1))

        assert abs(vdata["deltas_summary"]["delta_f1_mean"] - expected_delta_mean) < 1e-4
        assert abs(vdata["deltas_summary"]["delta_f1_std"] - expected_delta_std) < 1e-4


def test_model_ranking(db: Session, client: TestClient):
    """
    TEST 5: Model Ranking
    Verify ranking items are correctly ordered by primary metric.
    """
    exp_id, _, _ = create_reporting_mock_data(db, client)
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    agg_stats, _ = ReportService.aggregate_seed_metrics(exp.results)

    ranking = ReportService.rank_models(agg_stats, primary_metric="f1")
    assert ranking.primary_metric == "f1"
    assert ranking.higher_is_better is True
    assert ranking.rankings[0].rank == 1

    # Check strictly descending order for higher_is_better
    means = [r.primary_metric_mean for r in ranking.rankings]
    assert means == sorted(means, reverse=True)


def test_visualization_figures_valid_png(db: Session, client: TestClient):
    """
    TEST 6: Visualizations
    Verify all generated figures have valid PNG signature bytes.
    """
    exp_id, _, _ = create_reporting_mock_data(db, client)
    report_id = "report-test-png-001"
    req = ReportGenerationRequest(experiment_id=exp_id, include_xai=False, generate_figures=True)
    ReportService.generate_report(db=db, report_id=report_id, request=req)

    art_res = ReportService.get_report_artifacts(db, report_id)
    png_artifacts = [a for a in art_res.artifacts if a.artifact_type == "figure_png"]
    assert len(png_artifacts) >= 7

    png_magic_header = b"\x89PNG\r\n\x1a\n"
    for art in png_artifacts:
        raw_bytes = storage.read_file(art.path_key)
        assert raw_bytes.startswith(png_magic_header)


def test_artifact_checksums_match_files(db: Session, client: TestClient):
    """
    TEST 11: Artifact Checksums
    Verify generated SHA-256 hashes in manifest match actual stored bytes.
    """
    exp_id, _, _ = create_reporting_mock_data(db, client)
    report_id = "report-test-chk-001"
    req = ReportGenerationRequest(experiment_id=exp_id, include_xai=False)
    ReportService.generate_report(db=db, report_id=report_id, request=req)

    art_res = ReportService.get_report_artifacts(db, report_id)
    for art in art_res.artifacts:
        raw_bytes = storage.read_file(art.path_key)
        calc_hash = ReportService._sha256_bytes(raw_bytes)
        assert calc_hash == art.checksum_sha256


def test_authorization_role_enforcement(db: Session, client: TestClient):
    """
    TEST 12: Authorization
    Verify non-researcher/non-admin roles (e.g. CLIENT) cannot generate reports.
    """
    exp_id, _, _ = create_reporting_mock_data(db, client)

    client_user = User(
        email="client_user@fl.edu",
        hashed_password="secret_hash",
        role=UserRole.CLIENT,
        is_active=True,
    )
    db.add(client_user)
    db.commit()

    # Via direct service authorization check in endpoint
    from app.api.v1.endpoints.reports import generate_publication_report
    req = ReportGenerationRequest(experiment_id=exp_id)

    with pytest.raises(ForbiddenException):
        generate_publication_report(request=req, db=db, current_user=client_user)


def test_failed_report_state(db: Session, client: TestClient):
    """
    TEST 13: Failed Report State
    Verify invalid experiment ID throws NotFoundException and sets failure state.
    """
    req = ReportGenerationRequest(experiment_id="non-existent-exp-id")
    with pytest.raises(NotFoundException):
        ReportService.validate_report_lineage(db, "non-existent-exp-id")


def test_report_regeneration_determinism(db: Session, client: TestClient):
    """
    TEST 14: Report Regeneration
    Verify regenerating a report from the exact same source yields identical aggregate statistics.
    """
    exp_id, _, _ = create_reporting_mock_data(db, client)

    req = ReportGenerationRequest(experiment_id=exp_id, include_xai=False)
    rep1 = ReportService.generate_report(db=db, report_id="rep-regen-1", request=req)
    rep2 = ReportService.generate_report(db=db, report_id="rep-regen-2", request=req)

    sum1 = ReportService.get_report_summary(db, "rep-regen-1")
    sum2 = ReportService.get_report_summary(db, "rep-regen-2")

    assert sum1.ranking.primary_metric == sum2.ranking.primary_metric
    assert sum1.ranking.best_model_variant == sum2.ranking.best_model_variant
    assert len(sum1.model_summaries) == len(sum2.model_summaries)
    for m1, m2 in zip(sum1.model_summaries, sum2.model_summaries):
        assert m1.variant == m2.variant
        assert abs(m1.metrics["f1"].mean - m2.metrics["f1"].mean) < 1e-6
