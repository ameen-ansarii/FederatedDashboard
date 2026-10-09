from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.schemas.federated import FederatedClientResponse
from app.services.federated_client_service import FederatedClientService

router = APIRouter()


@router.get("/clients", response_model=List[FederatedClientResponse])
def list_federated_clients(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    clients = FederatedClientService.list_clients(db, skip=skip, limit=limit)
    return clients
