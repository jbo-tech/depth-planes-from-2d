# GCP logic implementation
import os
from google.cloud import storage


def get_data(path: str) -> list:
    """
    _summary_
    """

    if IMAGE_ENV == 'local':
        folder = "{}/{}".format(os.path.dirname(os.getcwd()),path)
        images = []
        for image in os.listdir(folder):
            images.append(folder + image)

    if IMAGE_ENV == 'gcp':
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_name)

        with blob.open("r") as f:

    return images

def save_data():
    """
    - Load data from the bucket and a specific bloob
    - ???
    """

    if IMAGE_ENV == 'local':


    if IMAGE_ENV == 'gcp':

        # Iterate on the bucket
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)

        for bucket_folder in bucket_folders:
            blobs = bucket.blob(bucket_folder)
            nb = 1

            # Create this folder locally if not exists
            folder = '{}/{}'.format(folder,bucket_folder)
            if not os.path.exists(folder):
                os.makedirs(folder)

            # Iterating through for loop one by one using API call
            for blob in blobs:
                logging.info('Blobs: {}'.format(blob.name))
                destination_uri = '{}/{}/{}'.format(folder,bucket_folder,blob.name)
                blob.download_to_filename(destination_uri)
                logging.info('Exported {} to {}'.format(blob.name, destination_uri))
                nb += 1
                if nb == nb_blobs:
                    break

if __name__ == '__main__':
    get_data('raw_data')
    #save_data()
