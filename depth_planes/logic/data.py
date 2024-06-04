# GCP logic implementation
from depth_planes.params import *
import os
import numpy as np
import datetime
from google.cloud import storage

def get_data(path: str) -> list:
    """
    _summary_
    """

    images = []

    if IMAGE_ENV == 'local':
        folder = "{}/{}".format(os.path.dirname(os.path.dirname(os.getcwd())),path)
        for image in os.listdir(folder):
            images.append(folder + image)

    if IMAGE_ENV == 'gcp':
        print('test')
        client = storage.Client(project=GCP_PROJECT)
        print(BUCKET_NAME)
        for blob in client.list_blobs(BUCKET_NAME, prefix=path):
            images.append(blob.name)
            #get_blob_url(blob)

    print(images)
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

        # Init GCP storage
        client = storage.Client(project=GCP_PROJECT)
        bucket = client.bucket(BUCKET_NAME)

        # Convert the file_array to bytes
        file_bytes = file_array.encode('utf-8')

        # Create a unique blob name
        blob_name = f"{path}/{name}.npy"

        # Upload the file to the bucket
        blob = bucket.blob(blob_name)
        blob.upload_from_string(file_bytes, content_type='application/octet-stream')

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

    print("Generated GET signed URL:")
    print(url)

    return url

if __name__ == '__main__':
    #get_data('urbansyn/rgb')
    #save_data('[[1,1][1,1]]','test','doss/perferct')
    #get_blob_url('urbansyn/rgb/rgb_7539.png')
