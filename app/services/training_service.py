from typing import Any, Dict, Optional, Tuple
from sqlalchemy.orm import Session

from app.core.reproducibility import build_reproducibility_metadata
from app.ml.trainer import CentralizedTrainer
from app.models.domain import JobStatus, ModelRegistry, TrainingRun, User
from app.schemas.training import CentralizedTrainingConfig
from app.services.model_service import ModelService
from app.services.preprocessing_service import PreprocessingService


class TrainingService:
    @staticmethod
    def run_centralized_training(
        db: Session,
        run_id: str,
        config: CentralizedTrainingConfig,
        user: Optional[User] = None,
    ) -> Tuple[ModelRegistry, TrainingRun]:
        # 1. Fetch exact preprocessing artifact from Phase 2
        artifact = PreprocessingService.get_artifact(db, config.artifact_id)
        preprocessor, splits = PreprocessingService.load_preprocessor_and_splits(artifact)

        input_dim = int(splits["X_train"].shape[1])

        # 2. Instantiate Centralized Trainer
        trainer = CentralizedTrainer(
            input_dim=input_dim,
            hidden_dims=config.hidden_dims,
            dropout_rate=config.dropout_rate,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            epochs=config.epochs,
            optimizer_name=config.optimizer,
            random_state=config.random_state,
        )

        # 3. Train and Evaluate on Test Set
        model, metrics, history = trainer.train_and_evaluate(splits)

        # 4. Save Model to Registry & Storage
        hyperparameters = config.model_dump()
        hyperparameters["reproducibility"] = build_reproducibility_metadata(
            base_random_state=config.base_random_state or config.random_state,
            derived_seed=config.random_state,
            algorithm="CENTRALIZED",
            variant="BASELINE",
        )
        model_record = ModelService.save_model(
            db=db,
            name=config.name,
            model=model,
            artifact=artifact,
            hyperparameters=hyperparameters,
            metrics=metrics,
            user=user,
        )

        # 5. Record Training Run entry
        training_run = TrainingRun(
            id=run_id,
            type="CENTRALIZED",
            status=JobStatus.SUCCESS,
            hyperparameters=hyperparameters,
            metrics=metrics,
            history=history,
            user_id=user.id if user else None,
        )
        db.add(training_run)
        db.commit()
        db.refresh(training_run)

        return model_record, training_run

    @staticmethod
    def get_training_run(db: Session, run_id: str) -> TrainingRun:
        run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        return run
