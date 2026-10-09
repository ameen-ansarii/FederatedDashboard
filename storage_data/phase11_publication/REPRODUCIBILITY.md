# Experimental Reproducibility Protocol

**Project:** Privacy-Preserving Federated Learning for Student Dropout Prediction with Explainable AI  
**Audited Benchmark Artifact:** `v_1_1786778298`  
**Base Seeds:** `[42, 123, 456, 789, 1024]` ($N=5$)  

---

## 1. Prerequisites & Environment Setup

Clone the repository and activate the isolated virtual environment:

```bash
cd "Privacy-Preserving Federated Learning Student Dropout Prediction"
source venv/bin/activate
pip install -r requirements.txt
```

Verify the environment:
- Python 3.13+
- PyTorch 2.x
- NumPy 2.x
- SciPy 1.17+
- Scikit-Learn 1.9+

---

## 2. Dataset & Preprocessing Lineage Verification

The benchmark evaluates the frozen audited OULAD preprocessing artifact `v_1_1786778298`:
- Location: `storage_data/artifacts/preprocessing/v_1_1786778298/`
- `splits.npz`: Contains `X_train` (120x33), `y_train` (120), `X_val` (40x33), `y_val` (40), `X_test` (40x33), `y_test` (40).
- `preprocessor.joblib`: Serialized ColumnTransformer.

Verify split matrices and test distribution:
```bash
python3 -c "
import numpy as np
splits = np.load('storage_data/artifacts/preprocessing/v_1_1786778298/splits.npz')
print('Test samples:', len(splits['y_test']), 'Class 0:', sum(splits['y_test']==0), 'Class 1:', sum(splits['y_test']==1))
"
```

---

## 3. Running the Multi-Seed Benchmark

Execute the complete 5-seed benchmark across all 6 model variants (30 models):

```bash
python3 scripts/run_phase9d_multiseed.py
```

This updates `storage_data/phase9d_multiseed_results.json` with exact seed trajectories, metrics, client distributions, and model weight checksums.

---

## 4. Executing the Scientific Evidence Audit (Phase 10B)

Run the independent scientific audit to recalculate descriptive metrics, confidence intervals, paired hypothesis tests, and effect sizes:

```bash
python3 -c "
from app.services.scientific_audit_service import ScientificAuditService
res = ScientificAuditService.run_full_audit()
print('Audit Status:', res['audit_status'])
"
```

---

## 5. Generating Publication Artifacts (Phase 11)

To regenerate all publication documentation, thesis chapters, figure registries, and evidence inventories:

```bash
python3 -c "
from app.services.publication_service import PublicationService
res = PublicationService.run_publication_generation()
print('Publication Status:', res['status'])
"
```

---

## 6. Running Automated Verification Tests

Run the full automated test suite (88+ tests) verifying zero fabrication, exact checksums, and mathematical correctness:

```bash
PYTHONPATH=. ./venv/bin/pytest -v tests/test_scientific_audit.py
PYTHONPATH=. ./venv/bin/pytest -v tests/test_phase11_publication.py
PYTHONPATH=. ./venv/bin/pytest -v
```
