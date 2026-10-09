from abc import ABC, abstractmethod
from typing import BinaryIO, Optional


class StorageBackend(ABC):
    @abstractmethod
    def save_file(self, file_path: str, contents: bytes) -> str:
        """Save bytes to storage path and return absolute location or reference."""
        pass

    @abstractmethod
    def read_file(self, file_path: str) -> bytes:
        """Read bytes from storage path."""
        pass

    @abstractmethod
    def delete_file(self, file_path: str) -> bool:
        """Delete file at storage path."""
        pass

    @abstractmethod
    def file_exists(self, file_path: str) -> bool:
        """Check if file exists at path."""
        pass
