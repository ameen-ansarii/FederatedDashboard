from typing import List
from sqlalchemy.orm import Session

from app.models.domain import FederatedClient


class FederatedClientService:
    @staticmethod
    def list_clients(db: Session, skip: int = 0, limit: int = 100) -> List[FederatedClient]:
        return db.query(FederatedClient).offset(skip).limit(limit).all()
