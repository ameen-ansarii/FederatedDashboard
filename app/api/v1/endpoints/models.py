from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.schemas.model import ModelDetailResponse, ModelRegistryResponse
from app.services.model_service import ModelService

router = APIRouter()


@router.get("", response_model=List[ModelRegistryResponse])
def list_models(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    models = ModelService.list_models(db, skip=skip, limit=limit)
    return models


@router.get("/{model_id}", response_model=ModelDetailResponse)
def get_model_detail(
    model_id: int,
    db: Session = Depends(get_db),
):
    model = ModelService.get_model(db, model_id)
    return model
