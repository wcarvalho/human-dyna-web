from google.cloud import storage
import os

from dotenv import load_dotenv

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
