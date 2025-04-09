import subprocess
import os
from google.cloud import storage
import fnmatch
from pathlib import Path

GOOGLE_CREDENTIALS = "./keys/datastore-key.json"


##############################
# User Data
##############################
def initialize_storage_client(bucket_name="human-dyna"):
  storage_client = storage.Client.from_service_account_json(GOOGLE_CREDENTIALS)
  bucket = storage_client.bucket(bucket_name)
  return bucket


def download_user_files(bucket_name, prefix, pattern, destination_folder):
  # Create a client
  bucket = initialize_storage_client(bucket_name)

  # List all blobs in the bucket with the given prefix
  blobs = bucket.list_blobs(prefix=prefix)

  # Create the destination folder if it doesn't exist
  os.makedirs(destination_folder, exist_ok=True)

  # Download matching files
  for blob in blobs:
    if fnmatch.fnmatch(blob.name, pattern):
      destination_file = os.path.join(destination_folder, os.path.basename(blob.name))
      # Check if file already exists
      if os.path.exists(destination_file):
        print(f"File already exists: \n\t {destination_file}")
        continue
      blob.download_to_filename(destination_file)
      print(f"Downloaded: \n\t from: {blob.name} \n\t to: {destination_file}")


if __name__ == "__main__":
  # housemaze
  bucket_name = "human-dyna"
  prefix = "data/"
  pattern = "*pilot-v1-r0-t0-plan*"
  pattern = f"data/data_user={pattern}"
  destination_folder = "/Users/wilka/git/research/results/human_dyna/user_data/exps"
  download_user_files(bucket_name, prefix, pattern, destination_folder)

  # craftax
