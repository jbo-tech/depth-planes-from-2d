# GCP logic implementation
from depth_planes.params import *
import os
import numpy as np
import datetime
from google.cloud import storage

def clean_data(bucket_name: str, path: str):
    """
    _summary_

    Args:
        bucket_name (str): _description_
        path (str): _description_
    """
    # Todo

def get_data(bucket_name: str, path: str) -> list:
    """
    _summary_
    """

    images = []

    if IMAGE_ENV == 'local':
        folder = "{}/{}".format(os.path.dirname(os.path.dirname(os.getcwd())),path)
        for image in os.listdir(folder):
            images.append(folder + image)

    if IMAGE_ENV == 'gcp':
        prefix = path
        images = list_files(bucket_name, extension=None, prefix=prefix)

    # print(images)
    return images

def save_data(file_array: str, name:str, path: str):
    """
    Save np.array to a npy file in a bucket and a specific folder
    """

    if IMAGE_ENV == 'local':
        # Check if the folder exists
        folder = "{}/{}".format(os.path.dirname(os.path.dirname(os.getcwd())),path)
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Create and save the npy file
        file_path = f'{folder}/{name}.npy'
        np.save(file_path, file_array)

    if IMAGE_ENV == 'gcp':
        prefix = path
        # Init GCP storage
        client = storage.Client(project=GCP_PROJECT)
        bucket = client.bucket(BUCKET_NAME)

        # Convert the file_array to bytes
        file_bytes = file_array.encode('utf-8')

        # Create a unique blob name
        blob_name = f"{prefix}/{name}.npy"

        # Upload the file to the bucket
        blob = bucket.blob(blob_name)
        blob.upload_from_string(file_bytes, content_type='application/octet-stream')

def list_files(bucket_name: str, extension=None, prefix=None):
    """
    Summary
    """
    #
    client = storage.Client(project=GCP_PROJECT)
    bucket = client.bucket(bucket_name)

    #
    blobs = bucket.list_blobs(prefix=prefix)

    #
    if extension:
        files = [blob.name for blob in blobs if blob.name.endswith(extension)]
    else:
        files = [blob.name for blob in blobs]

    #print(files)
    return files

def get_blob_url(blob_prefix: str) -> str:

    client = storage.Client(project=GCP_PROJECT)
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.get_blob(blob_prefix)

    url = blob.generate_signed_url(
        version="v4",
        # This URL is valid for 15 minutes
        expiration=datetime.timedelta(minutes=30),
        # Allow GET requests using this URL.
        method="GET",
    )

    #print("Generated GET signed URL:" + url)
    return url

if __name__ == '__main__':
    get_data(bucket_name='depth-planes-from-2d-data',path='make3d/test/depth')
    #save_data('[[1,1][1,1]]','test','doss/perferct')
    #get_blob_url('urbansyn/rgb/rgb_7539.png')
    #list_files(bucket_name='depth-planes-from-2d-data', extension=None, prefix='make3d/test/depth')
