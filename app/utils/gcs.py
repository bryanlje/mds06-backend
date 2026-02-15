################################################################################
# Google Cloud Storage helpers.
################################################################################

import logging
import re
from pathlib import Path
from google.cloud import storage as gcs

logger = logging.getLogger(__name__)

def _gcs():
    """Initializes the GCS Client."""
    return gcs.Client()

def _parse_gcs_uri(gcs_uri: str) -> tuple:
    """Parses a GCS URI into (bucket, blob_name)."""
    match = re.match(r"gs://([^/]+)/(.+)", gcs_uri)
    if not match:
        raise ValueError(f"Invalid GCS URI format: {gcs_uri}")
    return match.groups()

def upload_blob(local_path: Path, bucket_name: str, blob_name: str):
    """Uploads a file and returns its GCS URI."""
    try:
        blob = _gcs().bucket(bucket_name).blob(blob_name)
        blob.upload_from_filename(str(local_path))
        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        logger.info("Uploaded %s -> %s", local_path, gcs_uri)
        return gcs_uri
    except Exception as e:
        logger.error(f"Failed to upload {local_path}: {e}")
        return None

def download_blob(gcs_uri: str, local_path: str):
    """Downloads a blob from GCS to a local file."""
    try:
        bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
        blob = _gcs().bucket(bucket_name).blob(blob_name)
        
        if not blob.exists():
            raise FileNotFoundError(f"Blob not found: {gcs_uri}")

        blob.download_to_filename(local_path)
        logger.info(f"Downloaded {gcs_uri} -> {local_path}")
    except Exception as e:
        logger.exception(f"Failed to download {gcs_uri}")
        raise e