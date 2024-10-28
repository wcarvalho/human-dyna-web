from google.auth.exceptions import TransportError
from google.cloud import storage
import os
import json

from dotenv import load_dotenv

from google.cloud.exceptions import exceptions as gcs_exceptions
from nicewebrl.logging import get_logger


logger = get_logger(__name__)
load_dotenv()


def initialize_storage_client():
    storage_client = storage.Client.from_service_account_json(
        os.environ['GOOGLE_CREDENTIALS'])
    bucket_name = 'human-dyna'
    bucket = storage_client.bucket(bucket_name)
    return bucket


def list_files(bucket):
    blobs = bucket.list_blobs()
    print("Files in bucket:")
    for blob in blobs:
        print(blob.name)


def download_files(bucket, destination_folder):
    blobs = bucket.list_blobs()
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for blob in blobs:
        file_path = os.path.join(destination_folder, blob.name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        blob.download_to_filename(file_path)
        print(f"Downloaded {blob.name} to {file_path}")

def main():
    bucket = initialize_storage_client()

    ## List files in the bucket
    list_files(bucket)

    # Download files from the bucket
    #download_files(bucket, 'google_cloud_data')

if __name__ == "__main__":
    main()


async def save_data_to_gcs(data, blob_filename):
    try:
        bucket = initialize_storage_client()
        blob = bucket.blob(blob_filename)
        blob.upload_from_string(data=json.dumps(data), content_type='application/json')
        logger.info(f'Saved {blob_filename} in bucket {bucket.name}')
        return True  # Successfully saved
    except (TransportError, gcs_exceptions.GoogleCloudError) as e:
        logger.info(f"Error saving to GCS: {e}")
    except Exception as e:
        logger.info(f"Unexpected error: {e}")
        logger.info("Skipping GCS upload")

    return False  # Failed to save


async def save_file_to_gcs(local_filename, blob_filename):
    try:
        bucket = initialize_storage_client()
        blob = bucket.blob(blob_filename)
        blob.upload_from_filename(local_filename)
        logger.info(f'Saved {blob_filename} in bucket {bucket.name}')
        return True  # Successfully saved
    except (TransportError, gcs_exceptions.GoogleCloudError) as e:
        logger.info(f"Error saving to GCS: {e}")
    except Exception as e:
        logger.info(f"Unexpected error: {e}")
        logger.info("Skipping GCS upload")

    return False  # Failed to save
