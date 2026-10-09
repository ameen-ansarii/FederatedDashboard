import io
import json
import os
import uuid
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sqlalchemy.orm import Session

from app.core.errors import BadRequestException, NotFoundException
from app.models.domain import Dataset, PreprocessingArtifact, User
from app.schemas.preprocessing import PreprocessingConfig
from app.services.dataset_service import DatasetService
from app.storage.local import storage


class PreprocessingService:
    @staticmethod
    def preprocess_dataset(
        db: Session,
        dataset_id: int,
        config: PreprocessingConfig,
        user: Optional[User] = None,
    ) -> PreprocessingArtifact:
        # 1. Fetch dataset from DB
        dataset = DatasetService.get_dataset_by_id(db, dataset_id)
        df = DatasetService.get_dataset_dataframe(dataset)

        if df.empty:
            raise BadRequestException("Dataset is empty and cannot be preprocessed")

        # 1. Validate target column presence
        target_col = config.target_column or dataset.target_column
        if target_col not in df.columns:
            raise BadRequestException(
                f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}"
            )

        # 2. Handle duplicates
        if config.handle_duplicates:
            df = df.drop_duplicates().reset_index(drop=True)

        # Drop explicit columns if specified
        if config.drop_columns:
            cols_to_drop = [c for c in config.drop_columns if c in df.columns and c != target_col]
            df = df.drop(columns=cols_to_drop)

        # 3. Target mapping (Non-learned cleanup)
        y_raw = df[target_col]
        X_raw = df.drop(columns=[target_col])

        # Map target values to 0 and 1
        unique_targets = y_raw.unique()
        target_mapping = {}
        if config.positive_class_val is not None and config.positive_class_val in unique_targets:
            pos_val = config.positive_class_val
            target_mapping = {str(val): 1 if val == pos_val else 0 for val in unique_targets}
        else:
            # Automatic heuristic: check strings like "dropout", 1, True
            target_mapping = {}
            for val in unique_targets:
                str_val = str(val).strip().lower()
                if str_val in ["dropout", "1", "true", "yes", "1.0", "withdrawn"]:
                    target_mapping[str(val)] = 1
                else:
                    target_mapping[str(val)] = 0
            
            # If all mapped to 0 or 1, ensure at least one 1 and one 0 if 2 classes exist
            if len(set(target_mapping.values())) == 1 and len(unique_targets) == 2:
                target_mapping = {str(unique_targets[0]): 0, str(unique_targets[1]): 1}

        y = y_raw.astype(str).map(target_mapping).values

        # 4. Train / Validation / Test Split BEFORE fitting learned transformers!
        # Total non-train size = val_size + test_size
        temp_size = config.val_size + config.test_size
        if temp_size >= 1.0 or temp_size <= 0.0:
            raise BadRequestException("val_size + test_size must be strictly between 0.0 and 1.0")

        # Stratified split if binary
        stratify_y = y if len(np.unique(y)) > 1 else None

        X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
            X_raw, y, test_size=temp_size, random_state=config.random_state, stratify=stratify_y
        )

        relative_test_ratio = config.test_size / temp_size
        stratify_temp = y_temp if len(np.unique(y_temp)) > 1 else None

        X_val_raw, X_test_raw, y_val, y_test = train_test_split(
            X_temp_raw, y_temp, test_size=relative_test_ratio, random_state=config.random_state, stratify=stratify_temp
        )

        # 5. Identify Numerical and Categorical feature columns
        num_cols = list(X_train_raw.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns)
        cat_cols = [c for c in X_train_raw.columns if c not in num_cols]

        transformers = []
        if num_cols:
            num_steps = [("imputer", SimpleImputer(strategy=config.impute_strategy))]
            if config.scale_features:
                num_steps.append(("scaler", StandardScaler()))
            transformers.append(("num", Pipeline(num_steps), num_cols))

        if cat_cols:
            cat_steps = [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
            transformers.append(("cat", Pipeline(cat_steps), cat_cols))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="passthrough")

        # FIT TRANSFORMER ON TRAINING DATA ONLY!
        X_train = preprocessor.fit_transform(X_train_raw)

        # TRANSFORM VALIDATION AND TEST SPLITS WITHOUT REFITTING!
        X_val = preprocessor.transform(X_val_raw)
        X_test = preprocessor.transform(X_test_raw)

        # Extract output feature names
        feature_names = []
        if num_cols:
            feature_names.extend(num_cols)
        if cat_cols:
            cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
            encoded_cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
            feature_names.extend(encoded_cat_names)

        # 6. Save artifacts to storage
        version_str = f"v_{dataset_id}_{int(pd.Timestamp.now(tz='UTC').timestamp())}_{uuid.uuid4().hex[:6]}"
        artifact_dir = f"artifacts/preprocessing/{version_str}"

        # Save fitted transformer pipeline
        pipeline_bytes_io = io.BytesIO()
        joblib.dump(preprocessor, pipeline_bytes_io)
        pipeline_bytes = pipeline_bytes_io.getvalue()
        pipeline_rel_path = f"{artifact_dir}/preprocessor.joblib"
        storage.save_file(pipeline_rel_path, pipeline_bytes)

        # Save split datasets
        split_data_io = io.BytesIO()
        np.savez_compressed(
            split_data_io,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
        )
        splits_rel_path = f"{artifact_dir}/splits.npz"
        storage.save_file(splits_rel_path, split_data_io.getvalue())

        feature_schema = {
            "feature_names": feature_names,
            "feature_count": len(feature_names),
            "numerical_features": num_cols,
            "categorical_features": cat_cols,
            "scaled": config.scale_features,
            "impute_strategy": config.impute_strategy,
        }

        split_summary = {
            "train_samples": int(X_train.shape[0]),
            "val_samples": int(X_val.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "positive_class_ratio_train": float(np.mean(y_train)),
            "positive_class_ratio_val": float(np.mean(y_val)),
            "positive_class_ratio_test": float(np.mean(y_test)),
        }

        # Save record to DB
        artifact = PreprocessingArtifact(
            dataset_id=dataset_id,
            version=version_str,
            artifact_path=artifact_dir,
            feature_schema=feature_schema,
            target_mapping=target_mapping,
            split_summary=split_summary,
            user_id=user.id if user else None,
        )
        db.add(artifact)
        db.commit()
        db.refresh(artifact)
        return artifact

    @staticmethod
    def get_artifact(db: Session, artifact_id: int) -> PreprocessingArtifact:
        artifact = db.query(PreprocessingArtifact).filter(PreprocessingArtifact.id == artifact_id).first()
        if not artifact:
            raise NotFoundException(f"Preprocessing artifact with ID {artifact_id} not found")
        return artifact

    @staticmethod
    def load_preprocessor_and_splits(artifact: PreprocessingArtifact) -> Tuple[Any, Dict[str, np.ndarray]]:
        pipeline_bytes = storage.read_file(f"{artifact.artifact_path}/preprocessor.joblib")
        preprocessor = joblib.load(io.BytesIO(pipeline_bytes))

        splits_bytes = storage.read_file(f"{artifact.artifact_path}/splits.npz")
        npz = np.load(io.BytesIO(splits_bytes), allow_pickle=True)
        splits = {
            "X_train": npz["X_train"],
            "y_train": npz["y_train"],
            "X_val": npz["X_val"],
            "y_val": npz["y_val"],
            "X_test": npz["X_test"],
            "y_test": npz["y_test"],
            "feature_names": list(npz["feature_names"]),
        }
        return preprocessor, splits
