"""
Phase 11 — Research Paper, Thesis & Publication Artifact Validation Tests.

Validates that all publication artifacts, machine-readable results, claims, figure registries,
and documentation are mathematically consistent, uncorrupted, and accurately grounded in verified
underlying experimental artifacts with zero metric fabrication.
"""

import csv
import hashlib
import json
import os
import re
from typing import Any, Dict, List

import numpy as np
import pytest

from app.core.config import settings
from app.services.publication_service import PublicationService
from app.services.scientific_audit_service import ScientificAuditService


@pytest.fixture(scope="module")
def pub_dir():
    d = os.path.join(settings.STORAGE_DIR, "phase11_publication")
    if not os.path.exists(d):
        PublicationService.run_publication_generation(d)
    return d


@pytest.fixture(scope="module")
def raw_data():
    path = os.path.join(settings.STORAGE_DIR, "phase9d_multiseed_results.json")
    if not os.path.exists(path):
        pytest.skip("Phase 9D multi-seed results not found.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def pub_results(pub_dir):
    p = os.path.join(pub_dir, "publication_results.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def registry(pub_dir):
    p = os.path.join(pub_dir, "publication_registry.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def inventory(pub_dir):
    p = os.path.join(pub_dir, "evidence_inventory.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def claim_audit_rows(pub_dir):
    p = os.path.join(pub_dir, "claim_audit.csv")
    with open(p, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def research_paper_text(pub_dir):
    p = os.path.join(pub_dir, "research_paper.md")
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def thesis_text(pub_dir):
    p = os.path.join(pub_dir, "thesis_chapters.md")
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


# ============================================================================
# 1. No Fabricated Metrics & Source Fidelity Tests
# ============================================================================

class TestMetricFidelityAndNoFabrication:
    def test_aggregate_metrics_match_raw_source(self, raw_data, pub_results):
        """Verify aggregate metrics in publication_results.json match raw phase9d exactly."""
        raw_agg = raw_data["aggregated_statistics"]
        pub_agg = pub_results["aggregate_metrics"]

        for vname, vdata in raw_agg.items():
            assert vname in pub_agg, f"Missing variant {vname} in publication results"
            raw_ms = vdata["metrics_summary"]
            pub_ms = pub_agg[vname]["metrics_summary"]

            for mk in ["accuracy_mean", "f1_mean", "roc_auc_mean", "loss_mean"]:
                assert abs(raw_ms[mk] - pub_ms[mk]) < 1e-6, \
                    f"Discrepancy in {vname} {mk}: {raw_ms[mk]} != {pub_ms[mk]}"

    def test_paper_mentions_exact_centralized_f1(self, raw_data, research_paper_text):
        """Verify research paper reports exact Centralized mean F1 (0.4971)."""
        cent_f1 = raw_data["aggregated_statistics"]["CENTRALIZED (BASELINE)"]["metrics_summary"]["f1_mean"]
        formatted_f1 = f"{cent_f1:.4f}"
        assert formatted_f1 in research_paper_text, f"Expected {formatted_f1} in research_paper.md"

    def test_paper_mentions_exact_fedavg_iid_and_noniid_f1(self, raw_data, research_paper_text):
        """Verify research paper reports exact FedAvg IID (0.4386) and Non-IID (0.3672) F1."""
        iid_f1 = raw_data["aggregated_statistics"]["FEDAVG (IID)"]["metrics_summary"]["f1_mean"]
        noniid_f1 = raw_data["aggregated_statistics"]["FEDAVG (NON_IID_ALPHA_0.5)"]["metrics_summary"]["f1_mean"]
        assert f"{iid_f1:.4f}" in research_paper_text
        assert f"{noniid_f1:.4f}" in research_paper_text


# ============================================================================
# 2. Seed-Level Fidelity and Lineage Tests
# ============================================================================

class TestSeedAndLineageFidelity:
    def test_seed_list_matches_phase9d(self, pub_results):
        """Verify seeds list matches exact [42, 123, 456, 789, 1024]."""
        assert pub_results["seed_configuration"]["seeds"] == [42, 123, 456, 789, 1024]

    def test_model_configuration_fidelity(self, pub_results):
        """Verify model architecture and training hyperparameters match experimental design."""
        cfg = pub_results["model_configuration"]
        assert cfg["architecture"] == "DropoutMLP"
        assert cfg["input_dim"] == 33
        assert cfg["hidden_dims"] == [32, 16]
        assert cfg["dropout_rate"] == 0.1
        assert cfg["learning_rate"] == 0.01
        assert cfg["batch_size"] == 16
        assert cfg["centralized_epochs"] == 10
        assert cfg["federated_rounds"] == 5
        assert cfg["local_epochs"] == 2
        assert cfg["clients_count"] == 3

    def test_dataset_lineage_preservation(self, pub_results):
        """Verify dataset lineage matches audited artifact v_1_1786778298."""
        ds = pub_results["dataset_metadata"]
        assert ds["total_audited_samples"] == 200
        assert ds["train_samples"] == 120
        assert ds["val_samples"] == 40
        assert ds["test_samples"] == 40
        assert ds["test_class_0_non_dropout"] == 27
        assert ds["test_class_1_dropout"] == 13
        assert ds["transformed_features"] == 33
        assert pub_results["preprocessing_artifact"]["version"] == "v_1_1786778298"


# ============================================================================
# 3. Statistical Integrity & P-Value Verification
# ============================================================================

class TestStatisticalFidelity:
    def test_p_values_match_statistical_artifacts(self, pub_results, claim_audit_rows):
        """Verify paired t-test p-value for FedAvg Non-IID vs IID equals 0.1951."""
        c1_claim = [c for c in claim_audit_rows if c["claim_id"] == "C1"][0]
        assert c1_claim["p_value"] == "0.1951"
        assert c1_claim["value"] == "-0.0714"

    def test_fedprox_mu_collapse_detection_fidelity(self, pub_results):
        """Verify FedProx mu audit matches verified collapse observations."""
        mu_data = {r["mu"]: r for r in pub_results["fedprox_analysis"]}
        assert int(mu_data["0.01"]["collapse_count"]) == 0
        assert int(mu_data["0.001"]["collapse_count"]) > 0
        assert int(mu_data["0.1"]["collapse_count"]) > 0


# ============================================================================
# 4. Publication Figures and Registry Validation
# ============================================================================

class TestPublicationRegistryAndFigures:
    def test_registry_contains_8_figures_and_7_tables(self, registry):
        """Verify registry maps 8 figures and 7 tables."""
        assert len(registry["figures"]) == 8
        assert len(registry["tables"]) >= 7

    def test_all_8_figures_exist_on_disk(self, pub_dir, registry):
        """Verify all 8 figure PNGs exist on disk and have non-zero size."""
        for fig in registry["figures"]:
            fpath = os.path.join(pub_dir, "figures", fig["filename"])
            assert os.path.exists(fpath), f"Figure {fig['filename']} missing"
            assert os.path.getsize(fpath) > 10000, f"Figure {fig['filename']} too small/empty"

    def test_figure_checksums_match_actual_files(self, pub_dir, registry):
        """Verify figure SHA-256 hashes in registry match actual disk files."""
        for fig in registry["figures"]:
            fpath = os.path.join(pub_dir, "figures", fig["filename"])
            with open(fpath, "rb") as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()
            assert actual_hash == fig["sha256"], f"Checksum mismatch for {fig['filename']}"


# ============================================================================
# 5. Claim Audit & Scientific Wording Validation
# ============================================================================

class TestClaimAuditAndWording:
    def test_claim_audit_contains_7_claims(self, claim_audit_rows):
        """Verify claim audit has claims C1 through C7."""
        claim_ids = [c["claim_id"] for c in claim_audit_rows]
        assert claim_ids == ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    def test_no_proves_or_universal_optimal_in_claims(self, claim_audit_rows):
        """Verify no claim asserts 'proves' or 'universal optimal'."""
        for c in claim_audit_rows:
            wording = c["allowed_wording"].lower()
            assert "proves" not in wording
            assert "universal optimal" not in wording
            assert "universally optimal" not in wording


# ============================================================================
# 6. Evidence Inventory & Artifact Manifest Integrity
# ============================================================================

class TestEvidenceInventoryAndManifest:
    def test_evidence_inventory_all_artifacts_available(self, inventory):
        """Verify all 15 audited evidence source artifacts are AVAILABLE (zero missing)."""
        assert inventory["missing_count"] == 0
        assert inventory["available_count"] == inventory["total_artifacts_audited"]
        assert inventory["available_count"] >= 15

    def test_publication_manifest_checksums(self, pub_dir):
        """Verify publication manifest correctly hashes all generated files."""
        manifest_path = os.path.join(pub_dir, "manifests", "publication_manifest.json")
        assert os.path.exists(manifest_path)

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for item in manifest["files"]:
            fpath = os.path.join(pub_dir, item["relative_path"])
            assert os.path.exists(fpath), f"Manifest file missing: {item['relative_path']}"
            with open(fpath, "rb") as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()
            assert actual_hash == item["checksum_sha256"], \
                f"Checksum mismatch for {item['relative_path']}"


# ============================================================================
# 7. Document Structure & Completeness Validation
# ============================================================================

class TestDocumentStructureAndCompleteness:
    def test_research_paper_has_23_sections(self, research_paper_text):
        """Verify research paper markdown contains all 23 standard sections."""
        required_headers = [
            "## Abstract",
            "## 1. Introduction",
            "## 2. Research Objectives and Questions",
            "## 3. Related Concepts",
            "## 4. Methodology & Dataset Specification",
            "## 5. Data Preprocessing & Lineage",
            "## 6. Model Architecture & Hyperparameters",
            "## 7. Federated Partitioning & Client Heterogeneity",
            "## 8. Reproducibility Protocol",
            "## 9. Experimental Design",
            "## 10. Statistical Analysis Framework",
            "## 11. Experimental Results",
            "## 12. Non-IID Client Heterogeneity Analysis",
            "## 13. FedProx mu Sensitivity & Minority-Class Collapse",
            "## 14. Multi-Seed Stability vs. Single-Seed Pilot",
            "## 15. Explainable AI (XAI) Attribution Consensus",
            "## 16. Explainability Limitations & Guardrails",
            "## 17. Computational Reproducibility & Integrity",
            "## 18. Scientific Limitations Registry",
            "## 19. Threats to Validity",
            "## 20. Discussion",
            "## 21. Conclusion",
            "## 22. Future Work",
            "## 23. Evidence Appendix",
        ]
        for h in required_headers:
            assert h in research_paper_text, f"Missing section in research paper: {h}"

    def test_thesis_has_11_chapters(self, thesis_text):
        """Verify thesis document contains Chapters 1 through 11."""
        for i in range(1, 12):
            assert f"# Chapter {i} —" in thesis_text, f"Missing Chapter {i} in thesis"

    def test_reproducibility_readme_contains_commands(self, pub_dir):
        """Verify REPRODUCIBILITY.md contains executable verification commands."""
        readme_path = os.path.join(pub_dir, "REPRODUCIBILITY.md")
        assert os.path.exists(readme_path)
        with open(readme_path, "r", encoding="utf-8") as f:
            text = f.read()
        assert "run_phase9d_multiseed.py" in text
        assert "ScientificAuditService.run_full_audit" in text
        assert "PublicationService.run_publication_generation" in text
