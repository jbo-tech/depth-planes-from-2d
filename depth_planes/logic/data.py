# GCP logic implementation
from depth_planes.params import *
import os
from google.cloud import storage

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"]='/home/jbo/code/soapoperator/gcp/neon-fiber-420411-dd5c2fc94b59.json'

def get_data(path: str) -> list:
    """
    _summary_
    """

    if IMAGE_ENV == 'local':
        folder = "{}/{}".format(os.path.dirname(os.path.dirname(os.getcwd())),path)
        images = []
        for image in os.listdir(folder):
            images.append(folder + image)

    if IMAGE_ENV == 'gcp':
        client = storage.Client(project=GCP_PROJECT)
        for blob in client.list_blobs(BUCKET_NAME, prefix=path):
            print(str(blob))
            # blob.download_to_filename(destination_file_name)

    print(images)
    return images

def save_data(file_array,
    name:str,
    path: str):
    """
    - Load data from the bucket and a specific bloob
    - ???
    """

    if IMAGE_ENV == 'local':
        with open(f'{path}/{name}.npy', 'wb') as f:
            np.save(f,file)

    if IMAGE_ENV == 'gcp':

        # Iterate on the bucket
        client = storage.Client(project=GCP_PROJECT)
        bucket = client.bucket(BUCKET_NAME)

        destination_uri = '{}/{}'.format(bucket_folder,blob.name)
        blob.download_to_filename(destination_uri)

if __name__ == '__main__':
    get_data('raw_data')
    #save_data()
