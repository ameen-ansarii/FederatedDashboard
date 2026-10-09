from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.models.domain import User
from app.schemas.xai import (
    GlobalSHAPRequest,
    GlobalSHAPResponse,
    UnifiedXAIResponse,
    XAIRequest,
    XAIResponse,
)
from app.services.xai_service import XAIService

router = APIRouter()


@router.post("/shap", response_model=XAIResponse, status_code=status.HTTP_200_OK)
def explain_with_shap(
    request: XAIRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Computes local SHAP explanations for a student dropout prediction
    using background distribution from the model's training lineage.
    """
    return XAIService.explain_shap(
        db=db,
        prediction_id=request.prediction_id,
        model_id=request.model_id,
        features=request.features,
        user=current_user,
    )


@router.post("/shap/global", response_model=GlobalSHAPResponse, status_code=status.HTTP_200_OK)
def explain_global_shap(
    request: GlobalSHAPRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Computes global feature importance via mean absolute SHAP values across dataset samples.
    """
    return XAIService.explain_global_shap(
        db=db,
        model_id=request.model_id,
        max_samples=request.max_samples,
        user=current_user,
    )


@router.post("/lime", response_model=XAIResponse, status_code=status.HTTP_200_OK)
def explain_with_lime(
    request: XAIRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Computes local LIME explanations for a student dropout prediction.
    """
    return XAIService.explain_lime(
        db=db,
        prediction_id=request.prediction_id,
        model_id=request.model_id,
        features=request.features,
        user=current_user,
    )


@router.post("/integrated-gradients", response_model=XAIResponse, status_code=status.HTTP_200_OK)
def explain_with_integrated_gradients(
    request: XAIRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Computes Integrated Gradients feature attributions using Captum against the PyTorch DropoutMLP.
    """
    return XAIService.explain_integrated_gradients(
        db=db,
        prediction_id=request.prediction_id,
        model_id=request.model_id,
        features=request.features,
        user=current_user,
    )


@router.get("/predictions/{prediction_id}", response_model=UnifiedXAIResponse, status_code=status.HTTP_200_OK)
def get_unified_prediction_explanation(
    prediction_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Returns a unified multi-method explainability report (SHAP, LIME, Integrated Gradients)
    for an existing saved prediction record.
    """
    return XAIService.get_unified_explanation(
        db=db,
        prediction_id=prediction_id,
        user=current_user,
    )
