import os
from google.cloud import storage


def parse_gcs_bucket_and_blob_name(gcs_path):
    splits = gcs_path.replace("gs://", "").split("/", 1)
    bucket = splits[0]
    blob_name = "" if len(splits) == 1 else splits[1]
    return bucket, blob_name


def get_blob(gcs_path):
    bucket, blob_name = parse_gcs_bucket_and_blob_name(gcs_path)
    assert blob_name, f"{blob_name=} should be a valid name"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket)
    blob = bucket.blob(blob_name)
    return blob


def get_file(path, mode):
    if path.startswith("gs://"):
        return get_blob(path).open(mode)
    else:
        file_dir = os.path.dirname(path)
        os.makedirs(file_dir, exist_ok=True)
        return open(path, mode)
