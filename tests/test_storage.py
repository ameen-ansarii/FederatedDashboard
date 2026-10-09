import tempfile
import os
from app.storage.local import LocalStorageBackend


def test_local_storage_operations():
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = LocalStorageBackend(base_dir=temp_dir)
        test_filename = "subfolder/test_data.txt"
        content = b"privacy_preserving_fl_test_content"

        # 1. Save
        saved_path = storage.save_file(test_filename, content)
        assert os.path.exists(saved_path)

        # 2. Exists
        assert storage.file_exists(test_filename) is True

        # 3. Read
        read_bytes = storage.read_file(test_filename)
        assert read_bytes == content

        # 4. Delete
        deleted = storage.delete_file(test_filename)
        assert deleted is True
        assert storage.file_exists(test_filename) is False
