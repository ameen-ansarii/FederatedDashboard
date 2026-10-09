"""
Phase 10B — Scientific Evidence Audit & Statistical Validation Tests.

Verifies the ScientificAuditService produces correct, reproducible, and
mathematically sound audit artifacts from Phase 9D multi-seed results.

Minimum required: 17 tests.
"""

import csv
import hashlib
import io
import json
import os
import tempfile
from typing import Any, Dict, List

import numpy as np
import pytest
import scipy.stats as stats

from app.services.scientific_audit_service import ScientificAuditService


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def raw_data():
    """Load the real Phase 9D multi-seed results once for the entire module."""
    path = os.path.join("storage_data", "phase9d_multiseed_results.json")
    if not os.path.exists(path):
        pytest.skip("Phase 9D multi-seed results not found; skipping audit tests.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def recalc_results(raw_data):
    return ScientificAuditService.recompute_metrics(raw_data)


@pytest.fixture(scope="module")
def ci_table(raw_data):
    return ScientificAuditService.compute_confidence_intervals(raw_data)


@pytest.fixture(scope="module")
def paired_tests(raw_data):
    return ScientificAuditService.compute_paired_statistical_comparisons(raw_data)


@pytest.fixture(scope="module")
def effect_sizes(raw_data):
    return ScientificAuditService.compute_effect_sizes(raw_data)


@pytest.fixture(scope="module")
def mu_audit(raw_data):
    return ScientificAuditService.audit_fedprox_mu(raw_data)


@pytest.fixture(scope="module")
def iid_audit(raw_data):
    return ScientificAuditService.audit_iid_vs_noniid(raw_data)


@pytest.fixture(scope="module")
def client_audit(raw_data):
    return ScientificAuditService.audit_client_heterogeneity(raw_data)


@pytest.fixture(scope="module")
def repro_audit(raw_data):
    return ScientificAuditService.audit_reproducibility(raw_data)


# ============================================================================
# Test 1: Raw artifact loading and structural verification
# ============================================================================

class TestRawArtifactVerification:
    def test_raw_artifact_verification_passes(self, raw_data):
        """T1: Verify that raw artifact verification passes with 30 models and 30 unique checksums."""
        result = ScientificAuditService.verify_raw_artifacts(raw_data)
        assert result["status"] == "PASSED"
        assert result["total_models_evaluated"] == 30
        assert result["expected_models"] == 30
        assert result["all_checksums_unique"] is True
        assert result["unique_checksum_count"] == 30
        assert result["unique_derived_seed_count"] == 30

    def test_raw_artifact_contains_correct_seeds(self, raw_data):
        """T2: Verify that exactly seeds [42, 123, 456, 789, 1024] are present."""
        assert raw_data["seeds"] == [42, 123, 456, 789, 1024]

    def test_raw_artifact_contains_all_six_variants(self, raw_data):
        """T3: Verify that all 6 variants appear in aggregated_statistics."""
        expected_variants = [
            "CENTRALIZED (BASELINE)",
            "FEDAVG (IID)",
            "FEDAVG (NON_IID_ALPHA_0.5)",
            "FEDPROX (NON_IID_ALPHA_0.5_MU_0.001)",
            "FEDPROX (NON_IID_ALPHA_0.5_MU_0.01)",
            "FEDPROX (NON_IID_ALPHA_0.5_MU_0.1)",
        ]
        agg_keys = list(raw_data["aggregated_statistics"].keys())
        for v in expected_variants:
            assert v in agg_keys, f"Missing variant: {v}"


# ============================================================================
# Test 4: Metric recalculation fidelity
# ============================================================================

class TestMetricRecalculation:
    def test_metric_recalculation_has_zero_discrepancies(self, recalc_results):
        """T4: Verify independent metric recalculation matches persisted values exactly."""
        assert recalc_results["audit_passed"] is True
        assert recalc_results["discrepancy_count"] == 0

    def test_recalculated_means_match_numpy(self, raw_data, recalc_results):
        """T5: Spot-check Centralized F1 mean against manual NumPy computation."""
        cent_runs = raw_data["aggregated_statistics"]["CENTRALIZED (BASELINE)"]["seed_runs"]
        f1_vals = [r["metrics"]["f1"] for r in cent_runs]
        expected_mean = float(np.mean(f1_vals))
        recalc_mean = recalc_results["recalculated_metrics"]["CENTRALIZED (BASELINE)"]["f1_mean"]
        assert abs(expected_mean - recalc_mean) < 1e-8


# ============================================================================
# Test 6-7: Confidence interval correctness
# ============================================================================

class TestConfidenceIntervals:
    def test_ci_count_equals_36(self, ci_table):
        """T6: 6 variants × 6 metrics = 36 CI rows."""
        assert len(ci_table) == 36

    def test_ci_bounds_bracket_mean(self, ci_table):
        """T7: Every CI lower bound is below the mean and CI upper bound is above."""
        for ci in ci_table:
            assert ci["ci_lower"] <= ci["mean"], f"{ci['variant']}/{ci['metric']}: CI lower {ci['ci_lower']} > mean {ci['mean']}"
            assert ci["ci_upper"] >= ci["mean"], f"{ci['variant']}/{ci['metric']}: CI upper {ci['ci_upper']} < mean {ci['mean']}"

    def test_ci_uses_correct_t_critical(self, ci_table):
        """T8: Verify all CIs use t_{0.975, 4} ≈ 2.7764."""
        expected_t = float(stats.t.ppf(0.975, 4))
        for ci in ci_table:
            assert abs(ci["t_critical"] - round(expected_t, 4)) < 0.001, \
                f"t_critical mismatch: {ci['t_critical']} vs {round(expected_t, 4)}"


# ============================================================================
# Test 9-10: Paired statistical tests
# ============================================================================

class TestPairedStatisticalTests:
    def test_paired_tests_count(self, paired_tests):
        """T9: 5 federated variants × 3 metrics (f1, accuracy, roc_auc) = 15 comparison rows."""
        assert len(paired_tests) == 15

    def test_paired_t_statistic_sign_for_noniid_f1(self, paired_tests):
        """T10: FedAvg Non-IID F1 should have negative t-statistic (degradation vs centralized)."""
        noniid_f1 = [p for p in paired_tests if "NON_IID_ALPHA_0.5)" in p["variant"] and p["metric"] == "f1"]
        assert len(noniid_f1) == 1
        assert noniid_f1[0]["t_statistic"] < 0, "FedAvg Non-IID should degrade F1 vs Centralized"

    def test_holm_bonferroni_correction_applied(self, paired_tests):
        """T11: Verify Holm-Bonferroni adjusted p-values are present and >= raw p-values for F1."""
        f1_tests = [p for p in paired_tests if p["metric"] == "f1"]
        for p in f1_tests:
            assert "holm_adjusted_p_value" in p
            assert p["holm_adjusted_p_value"] >= p["p_value_raw"] - 1e-8, \
                f"Holm p={p['holm_adjusted_p_value']} < raw p={p['p_value_raw']}"


# ============================================================================
# Test 12: Effect sizes
# ============================================================================

class TestEffectSizes:
    def test_effect_size_count(self, effect_sizes):
        """T12: 5 variants × 4 metrics (f1, accuracy, roc_auc, loss) = 20 effect size rows."""
        assert len(effect_sizes) == 20

    def test_hedges_g_has_bias_correction(self, effect_sizes):
        """T13: Hedges' g_z should differ from Cohen's d_z due to small-sample bias correction."""
        for es in effect_sizes:
            if es["cohens_dz"] != 0.0:
                assert es["hedges_gz"] != es["cohens_dz"], \
                    f"Hedges' g should differ from Cohen's d for {es['variant']}/{es['metric']}"
                # Hedges g should be slightly smaller in absolute value
                assert abs(es["hedges_gz"]) <= abs(es["cohens_dz"]) + 1e-8


# ============================================================================
# Test 14: FedProx mu audit and collapse detection
# ============================================================================

class TestFedProxMuAudit:
    def test_mu_audit_detects_collapse(self, mu_audit):
        """T14: Verify mu=0.001 and mu=0.1 have nonzero collapse counts, mu=0.01 has zero."""
        mu_map = {r["mu"]: r for r in mu_audit}
        assert mu_map["0.01"]["collapse_count"] == 0, "mu=0.01 should have no collapses"
        assert mu_map["0.001"]["collapse_count"] > 0, "mu=0.001 should have at least 1 collapse"
        assert mu_map["0.1"]["collapse_count"] > 0, "mu=0.1 should have at least 1 collapse"


# ============================================================================
# Test 15: Client heterogeneity
# ============================================================================

class TestClientHeterogeneity:
    def test_client_sample_sums_equal_120(self, client_audit):
        """T15: Every client partition (IID or Non-IID) should sum to 120 training samples."""
        for row in client_audit:
            assert row["sample_sum_valid"] is True, \
                f"Seed {row['seed']} {row['partition_type']}: total={row['total_samples']}, expected 120"


# ============================================================================
# Test 16: Test-set integrity
# ============================================================================

class TestTestSetIntegrity:
    def test_test_set_integrity(self):
        """T16: Held-out test split has exactly 40 samples (27 class-0, 13 class-1), 33 features."""
        result = ScientificAuditService.audit_test_set_integrity()
        assert result["passed"] is True
        assert result["test_sample_count"] == 40
        assert result["non_dropout_count"] == 27
        assert result["dropout_count"] == 13
        assert result["transformed_feature_dimension"] == 33


# ============================================================================
# Test 17: Claim-evidence matrix
# ============================================================================

class TestClaimEvidenceMatrix:
    def test_claim_evidence_matrix_has_7_claims(self, raw_data, recalc_results, paired_tests, effect_sizes):
        """T17: Claim-Evidence matrix contains exactly C1 through C7."""
        matrix = ScientificAuditService.build_claim_evidence_matrix(
            raw_data, recalc_results, paired_tests, effect_sizes,
        )
        claim_ids = [c["claim_id"] for c in matrix]
        assert claim_ids == ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
        for c in matrix:
            assert c["support_status"] == "SUPPORTED"


# ============================================================================
# Test 18: Reproducibility audit
# ============================================================================

class TestReproducibilityAudit:
    def test_all_30_checksums_unique(self, repro_audit):
        """T18: All 30 model checksums must be unique (zero collisions)."""
        assert repro_audit["unique_checksums"] == 30
        assert repro_audit["duplicate_checksum_count"] == 0
        assert repro_audit["reproducibility_status"] == "DETERMINISTIC_AND_ISOLATED"


# ============================================================================
# Test 19: Full audit pipeline end-to-end
# ============================================================================

class TestFullAuditPipeline:
    def test_full_audit_produces_13_artifacts(self, tmp_path):
        """T19: run_full_audit() generates 13 artifacts and a valid evidence manifest."""
        output_dir = str(tmp_path / "audit_output")
        result = ScientificAuditService.run_full_audit(output_dir=output_dir)

        assert result["audit_status"] == "PASSED"
        assert result["artifact_count"] == 13

        # Verify all expected files exist
        expected_files = [
            "scientific_audit_report.md",
            "metric_recalculation.json",
            "confidence_intervals.csv",
            "paired_tests.csv",
            "effect_sizes.csv",
            "fedprox_mu_audit.csv",
            "iid_vs_noniid_audit.csv",
            "client_heterogeneity_audit.csv",
            "xai_evidence_audit.csv",
            "figure_audit.csv",
            "claim_evidence_matrix.csv",
            "scientific_limitations.json",
            "evidence_manifest.json",
        ]
        for fname in expected_files:
            fpath = os.path.join(output_dir, fname)
            assert os.path.exists(fpath), f"Missing audit artifact: {fname}"
            assert os.path.getsize(fpath) > 0, f"Empty audit artifact: {fname}"

    def test_evidence_manifest_checksums_are_valid(self, tmp_path):
        """T20: Evidence manifest SHA-256 checksums match actual file contents."""
        output_dir = str(tmp_path / "audit_verify")
        ScientificAuditService.run_full_audit(output_dir=output_dir)

        manifest_path = os.path.join(output_dir, "evidence_manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        for entry in manifest["artifacts"]:
            fpath = entry["path"]
            expected_hash = entry["checksum_sha256"]
            with open(fpath, "rb") as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()
            assert actual_hash == expected_hash, \
                f"Checksum mismatch for {entry['filename']}: {actual_hash} != {expected_hash}"


# ============================================================================
# Test 21: Scientific limitations registry
# ============================================================================

class TestScientificLimitations:
    def test_limitations_registry_has_10_items(self):
        """T21: Limitations registry contains LIM-01 through LIM-10."""
        lims = ScientificAuditService.build_scientific_limitations()
        assert len(lims) == 10
        ids = [l["limitation_id"] for l in lims]
        for i in range(1, 11):
            assert f"LIM-{i:02d}" in ids


# ============================================================================
# Test 22: XAI evidence audit
# ============================================================================

class TestXAIEvidenceAudit:
    def test_xai_audit_returns_4_comparison_rows(self):
        """T22: XAI evidence audit produces 4 rows (SHAP-LIME, SHAP-IG, LIME-IG, Consensus)."""
        xai = ScientificAuditService.audit_xai_evidence()
        assert len(xai) == 4
        comparisons = [x["comparison"] for x in xai]
        assert "SHAP vs LIME" in comparisons
        assert "SHAP vs IG" in comparisons
        assert "LIME vs IG" in comparisons
        assert "Consensus (All 3 Methods)" in comparisons

    def test_xai_consensus_features_include_engagement(self):
        """T23: Consensus top features must include total_vle_clicks, avg_assessment_score, studied_credits."""
        xai = ScientificAuditService.audit_xai_evidence()
        consensus = [x for x in xai if x["comparison"] == "Consensus (All 3 Methods)"][0]
        expected_consensus = {"total_vle_clicks", "avg_assessment_score", "studied_credits"}
        assert expected_consensus.issubset(set(consensus["overlap_features"]))
