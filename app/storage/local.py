import os
import shutil
from pathlib import Path
from typing import Optional
from app.core.config import settings
from app.core.errors import NotFoundException
from app.storage.base import StorageBackend


class LocalStorageBackend(StorageBackend):
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or settings.STORAGE_DIR).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, relative_or_abs_path: str) -> Path:
        target_path = Path(relative_or_abs_path)
        if not target_path.is_absolute():
            target_path = self.base_dir / target_path
        return target_path.resolve()

    def save_file(self, file_path: str, contents: bytes) -> str:
        full_path = self._resolve_path(file_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "wb") as f:
            f.write(contents)
        return str(full_path)

    def read_file(self, file_path: str) -> bytes:
        full_path = self._resolve_path(file_path)
        if not full_path.exists():
            raise NotFoundException(f"File not found at storage location: {file_path}")
        with open(full_path, "rb") as f:
            return f.read()

    def delete_file(self, file_path: str) -> bool:
        full_path = self._resolve_path(file_path)
        if full_path.exists():
            if full_path.is_dir():
                shutil.rmtree(full_path)
            else:
                full_path.unlink()
            return True
        return False

    def file_exists(self, file_path: str) -> bool:
        full_path = self._resolve_path(file_path)
        return full_path.exists()


storage = LocalStorageBackend()
