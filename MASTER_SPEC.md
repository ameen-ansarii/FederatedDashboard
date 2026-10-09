# Master Project Specification

This document defines the complete requirements for:

Privacy-Preserving Federated Learning for Student Dropout Prediction with Explainable AI.

This is the authoritative human-readable specification.

The implementation target is a production-oriented REST API backend.

Frontend development is explicitly OUT OF SCOPE unless requested separately.

The backend must be ready for React/Next.js integration.

## Key System Modules & Phased Architecture
1. **Foundation**: FastAPI, PostgreSQL, SQLAlchemy, Alembic, Storage Abstraction, Structured Logging & Errors, Redis, Celery, Job tracking, Health monitoring, Docker setup.
2. **Dataset & Preprocessing**: Upload, validation, leakage-free preprocessing (fitting scalers/imputers on train split only), schema versioning, async job execution.
3. **Centralized ML**: PyTorch dropout prediction network, class-weighted training, metrics calculation (accuracy, precision, recall, f1, roc-auc), immutable model registry, lineage tracking.
4. **Real FedAvg**: Flower framework orchestration, IID and Non-IID Dirichlet partitioning, local client training, sample-weighted aggregation, privacy boundary enforcement (no raw data transfer to server).
5. **Real FedProx**: FedProx local objective optimization with proximal term penalty ($\frac{\mu}{2} \|w - w_{\text{global}}\|^2$), configurable $\mu$, frozen global reference.
6. **Prediction Engine**: Single and batch prediction pipelines, risk classification (LOW < 0.40, MEDIUM 0.40-0.70, HIGH >= 0.70), prediction history tracking.
7. **Explainable AI (XAI)**: SHAP, LIME, Captum (Integrated Gradients), JSON feature attributions, mandatory non-causal disclaimer, feature retention policies.
8. **Jobs & Live Progress**: Redis pub/sub, authenticated WebSockets (`/ws/training/{run_id}`), cooperative cancellation, retry mechanics, reconnect snapshots.
9. **Security**: JWT authentication, bcrypt password hashing, RBAC (`ADMIN`, `RESEARCHER`, `CLIENT`, `VIEWER`), ownership enforcement, CORS, rate limiting.
10. **Dashboard & Experiments**: Aggregated metrics, risk distribution, convergence charts, feature importance, experiment comparisons, model version management.
11. **Deployment & Final Validation**: Docker Compose production setup, automated test suite (unit, integration, end-to-end).
