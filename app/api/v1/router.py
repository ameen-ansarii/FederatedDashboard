from fastapi import APIRouter
from app.api.v1.endpoints import (
    health,
    datasets,
    training,
    models,
    fedavg,
    fedprox,
    federated_clients,
    predictions,
    xai,
    experiments,
    reports,
)

api_router = APIRouter()
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["Datasets & Preprocessing"])
api_router.include_router(training.router, prefix="/training", tags=["Centralized Training"])
api_router.include_router(fedavg.router, prefix="/training", tags=["FedAvg Orchestration"])
api_router.include_router(fedprox.router, prefix="/training", tags=["FedProx Orchestration"])
api_router.include_router(models.router, prefix="/models", tags=["Model Registry"])
api_router.include_router(federated_clients.router, prefix="/federated", tags=["Federated Clients"])
api_router.include_router(predictions.router, prefix="/predictions", tags=["Prediction Engine"])
api_router.include_router(xai.router, prefix="/xai", tags=["Explainable AI (XAI)"])
api_router.include_router(experiments.router, prefix="/experiments", tags=["Experimental Evaluation & Comparison"])
api_router.include_router(reports.router, prefix="/reports", tags=["Publication Reports & Artifacts"])




