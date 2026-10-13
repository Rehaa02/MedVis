import os
import shutil
from pathlib import Path
from typing import Optional, BinaryIO
import io

from backend.app.core.config import settings

class StorageManager:
    """
    Unified Object Storage Manager.
    Works natively with MinIO S3 API, and provides a zero-setup local filesystem
    fallback with identical object-store bucket semantics when MinIO service is offline.
    """
    def __init__(self):
        self.use_local = settings.USE_LOCAL_STORAGE
        self.local_root = settings.STORAGE_DIR
        self.bucket_name = settings.MINIO_BUCKET_NAME
        self.minio_client = None
        
        if not self.use_local:
            try:
                from minio import Minio
                self.minio_client = Minio(
                    settings.MINIO_ENDPOINT,
                    access_key=settings.MINIO_ACCESS_KEY,
                    secret_key=settings.MINIO_SECRET_KEY,
                    secure=settings.MINIO_SECURE
                )
                if not self.minio_client.bucket_exists(self.bucket_name):
                    self.minio_client.make_bucket(self.bucket_name)
            except Exception as e:
                print(f"[StorageManager] MinIO connection failed: {e}. Falling back to local object storage.")
                self.use_local = True

        if self.use_local:
            (self.local_root / self.bucket_name).mkdir(parents=True, exist_ok=True)

    def put_object(self, object_name: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Stores binary object data (DICOM pixel payloads or masks)."""
        if self.use_local or not self.minio_client:
            target_path = self.local_root / self.bucket_name / object_name
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "wb") as f:
                f.write(data)
            return str(target_path)
        else:
            data_stream = io.BytesIO(data)
            self.minio_client.put_object(
                self.bucket_name,
                object_name,
                data_stream,
                length=len(data),
                content_type=content_type
            )
            return f"s3://{self.bucket_name}/{object_name}"

    def put_file(self, object_name: str, file_path: str) -> str:
        """Copies an existing file into the object store."""
        with open(file_path, "rb") as f:
            data = f.read()
        return self.put_object(object_name, data)

    def get_object(self, object_name: str) -> bytes:
        """Retrieves binary object data."""
        if self.use_local or not self.minio_client:
            target_path = self.local_root / self.bucket_name / object_name
            if not target_path.exists():
                raise FileNotFoundError(f"Object {object_name} not found in local storage.")
            with open(target_path, "rb") as f:
                return f.read()
        else:
            response = self.minio_client.get_object(self.bucket_name, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data

    def get_local_path_if_exists(self, object_name: str) -> Optional[str]:
        target_path = self.local_root / self.bucket_name / object_name
        if target_path.exists():
            return str(target_path)
        return None

storage_manager = StorageManager()
