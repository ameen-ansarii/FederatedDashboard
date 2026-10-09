import hashlib
import io
import os
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from sqlalchemy.orm import Session

from app.core.errors import BadRequestException, NotFoundException
from app.models.domain import Dataset, User
from app.storage.local import storage


class DatasetService:
    @staticmethod
    def save_and_create_dataset(
        db: Session,
        name: str,
        filename: str,
        content: bytes,
        user: Optional[User] = None,
        target_column: str = "Target",
    ) -> Dataset:
        # Validate CSV format
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise BadRequestException(f"Invalid CSV file format: {str(e)}")

        if df.empty:
            raise BadRequestException("Uploaded CSV file is empty")

        row_count, col_count = df.shape
        checksum = hashlib.sha256(content).hexdigest()

        # Save to storage
        storage_rel_path = f"datasets/{checksum[:10]}_{filename}"
        saved_path = storage.save_file(storage_rel_path, content)

        dataset = Dataset(
            name=name,
            filename=filename,
            file_path=storage_rel_path,
            row_count=row_count,
            feature_count=col_count - 1 if target_column in df.columns else col_count,
            target_column=target_column if target_column in df.columns else (df.columns[-1] if len(df.columns) > 0 else "Target"),
            checksum=checksum,
            user_id=user.id if user else None,
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        return dataset

    @staticmethod
    def get_dataset_by_id(db: Session, dataset_id: int) -> Dataset:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise NotFoundException(f"Dataset with ID {dataset_id} not found")
        return dataset

    @staticmethod
    def get_dataset_dataframe(dataset: Dataset) -> pd.DataFrame:
        file_bytes = storage.read_file(dataset.file_path)
        return pd.read_csv(io.BytesIO(file_bytes))

    @staticmethod
    def get_dataset_detail(db: Session, dataset_id: int) -> Dict[str, Any]:
        dataset = DatasetService.get_dataset_by_id(db, dataset_id)
        df = DatasetService.get_dataset_dataframe(dataset)

        sample_rows = df.head(5).to_dict(orient="records")
        columns = list(df.columns)

        # Compute basic summary stats for numerical columns
        numeric_df = df.select_dtypes(include=["number"])
        summary_stats = numeric_df.describe().to_dict() if not numeric_df.empty else {}

        return {
            "id": dataset.id,
            "name": dataset.name,
            "filename": dataset.filename,
            "row_count": dataset.row_count,
            "feature_count": dataset.feature_count,
            "target_column": dataset.target_column,
            "checksum": dataset.checksum,
            "user_id": dataset.user_id,
            "created_at": dataset.created_at,
            "updated_at": dataset.updated_at,
            "columns": columns,
            "sample_rows": sample_rows,
            "summary_stats": summary_stats,
        }

    @staticmethod
    def delete_dataset(db: Session, dataset_id: int) -> bool:
        dataset = DatasetService.get_dataset_by_id(db, dataset_id)
        storage.delete_file(dataset.file_path)
        db.delete(dataset)
        db.commit()
        return True
