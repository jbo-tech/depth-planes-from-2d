# GCP logic implementation
from params import *
import os
import numpy as np
import datetime
from google.cloud import storage
import shutil
from pathlib import Path


def clean_data(path: str):
    """
    _summary_

    Args:
        bucket_name (str): _description_
        path (str): _description_
    """
    # files = glob.glob(path)
    # for f in files:
    #     os.remove(f)
    shutil.rmtree(path)


def get_npy(path: str) -> np.array:
    """
    _summary_

    Args:
        path (str): _description_

    Returns:
        np.array: _description_
    """
    list_npy = local_list_files(path)
    list_npy_array =[]
    for f in sorted(list_npy):
        npy = np.load(f)
        list_npy_array.append(npy)

    response = np.expand_dims(list_npy_array,axis=3) if np.array(list_npy_array).ndim == 3 else np.array(list_npy_array)
    #print(response)
    return response


def get_npy_direct(path_direct: str) -> np.array:
    """
    _summary_

    Args:
        path (str): _description_

    Returns:
        np.array: _description_
    """
    npy = np.load(path_direct)

    #print(response)
    return npy


def local_save_data(file_array: np.ndarray, name:str, path: str):
    """
    Save np.array to a npy file in a bucket and a specific folder
    """

    # Check if the folder exists
    if not os.path.exists(path):
        os.makedirs(path)

    # Create and save the npy file
    file_path = f'{path}/{name}.npy'
    #print(file_path)
    np.save(file_path, file_array)

    return file_path


def local_list_files(start_path='.') -> list:
    """
    _summary_

    Args:
        start_path (str): _description_

    Returns:
        list: _description_
    """
    list_files = []
    for root, dirs, files in os.walk(start_path):
        for file in files:
            #print(os.path.join(root, file))
            list_files.append(os.path.join(root, file))

    return list_files


def gcp_list_files(prefix=None,extension=None):
    """
    Summary
    """
    #
    client = storage.Client(project=GCP_PROJECT_OLD)
    bucket = client.bucket(BUCKET_NAME_OLD)

    #
    blobs = bucket.list_blobs(prefix=prefix)

    #
    if extension:
        files = [blob.name for blob in blobs if blob.name.endswith(extension)]
    else:
        files = [blob.name for blob in blobs]

    #print(files)
    return files


def download_many_blobs_with_transfer_manager(
    blob_names, destination_directory="", workers=8):
    """Download blobs in a list by name, concurrently in a process pool.

    The filename of each blob once downloaded is derived from the blob name and
    the `destination_directory `parameter. For complete control of the filename
    of each blob, use transfer_manager.download_many() instead.

    Directories will be created automatically as needed to accommodate blob
    names that include slashes.
    """

    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The list of blob names to download. The names of each blobs will also
    # be the name of each destination file (use transfer_manager.download_many()
    # instead to control each destination file name). If there is a "/" in the
    # blob name, then corresponding directories will be created on download.
    # blob_names = ["myblob", "myblob2"]

    # The directory on your computer to which to download all of the files. This
    # string is prepended (with os.path.join()) to the name of each blob to form
    # the full path. Relative paths and absolute paths are both accepted. An
    # empty string means "the current working directory". Note that this
    # parameter allows accepts directory traversal ("../" etc.) and is not
    # intended for unsanitized end user input.
    # destination_directory = ""

    # The maximum number of processes to use for the operation. The performance
    # impact of this value depends on the use case, but smaller files usually
    # benefit from a higher number of processes. Each additional process occupies
    # some CPU and memory resources until finished. Threads can be used instead
    # of processes by passing `worker_type=transfer_manager.THREAD`.
    # workers=8

    from google.cloud.storage import Client, transfer_manager

    storage_client = Client(project=GCP_PROJECT_OLD)
    bucket = storage_client.bucket(BUCKET_NAME_OLD)

    results = transfer_manager.download_many_to_path(
        bucket, blob_names, destination_directory=destination_directory, max_workers=workers
    )

    for name, result in zip(blob_names, results):
        # The results list is either `None` or an exception for each blob in
        # the input list, in order.

        if isinstance(result, Exception):
            print("Failed to download {} due to exception: {}".format(name, result))
        else:
            print("Downloaded {} to {}.".format(name, str(destination_directory) + '/' + name))


def download_chunks_concurrently(
    blob_name, filename, chunk_size=64 * 1024 * 1024, workers=8):
    """
    Download a single file in chunks, concurrently in a process pool.
    """

    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The file to be downloaded
    # blob_name = "target-file"

    # The destination filename or path
    # filename = ""

    # The size of each chunk. The performance impact of this value depends on
    # the use case. The remote service has a minimum of 5 MiB and a maximum of
    # 5 GiB.
    # chunk_size = 32 * 1024 * 1024 (32 MiB)

    # The maximum number of processes to use for the operation. The performance
    # impact of this value depends on the use case, but smaller files usually
    # benefit from a higher number of processes. Each additional process occupies
    # some CPU and memory resources until finished. Threads can be used instead
    # of processes by passing `worker_type=transfer_manager.THREAD`.
    # workers=8

    from google.cloud.storage import Client, transfer_manager

    storage_client = Client(project=GCP_PROJECT_OLD)
    bucket = storage_client.bucket(BUCKET_NAME_OLD)
    blob = bucket.blob(blob_name)

    transfer_manager.download_chunks_concurrently(
        blob, filename, chunk_size=chunk_size, max_workers=workers
    )

    print("Downloaded {} to {}.".format(blob_name, filename))


def upload_one_file(file_array: str, name:str, path: str):
    """
    Save np.array to a npy file in a bucket and a specific folder
    """
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

    return blob_name

def upload_many_blobs_with_transfer_manager(
    filenames, source_directory="", workers=8):
    """Upload every file in a list to a bucket, concurrently in a process pool.

    Each blob name is derived from the filename, not including the
    `source_directory` parameter. For complete control of the blob name for each
    file (and other aspects of individual blob metadata), use
    transfer_manager.upload_many() instead.
    """

    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # A list (or other iterable) of filenames to upload.
    # filenames = ["file_1.txt", "file_2.txt"]

    # The directory on your computer that is the root of all of the files in the
    # list of filenames. This string is prepended (with os.path.join()) to each
    # filename to get the full path to the file. Relative paths and absolute
    # paths are both accepted. This string is not included in the name of the
    # uploaded blob; it is only used to find the source files. An empty string
    # means "the current working directory". Note that this parameter allows
    # directory traversal (e.g. "/", "../") and is not intended for unsanitized
    # end user input.
    # source_directory=""

    # The maximum number of processes to use for the operation. The performance
    # impact of this value depends on the use case, but smaller files usually
    # benefit from a higher number of processes. Each additional process occupies
    # some CPU and memory resources until finished. Threads can be used instead
    # of processes by passing `worker_type=transfer_manager.THREAD`.
    # workers=8

    from google.cloud.storage import Client, transfer_manager

    storage_client = Client(project=GCP_PROJECT)
    bucket = storage_client.bucket(BUCKET_NAME)

    results = transfer_manager.upload_many_from_filenames(
        bucket, filenames, source_directory=source_directory, max_workers=workers
    )

    for name, result in zip(filenames, results):
        # The results list is either `None` or an exception for each filename in
        # the input list, in order.

        if isinstance(result, Exception):
            print("Failed to upload {} due to exception: {}".format(name, result))
        else:
            print("Uploaded {} to {}.".format(name, bucket.name))


def upload_directory_with_transfer_manager(source_directory, workers=8):
    """Upload every file in a directory, including all files in subdirectories.

    Each blob name is derived from the filename, not including the `directory`
    parameter itself. For complete control of the blob name for each file (and
    other aspects of individual blob metadata), use
    transfer_manager.upload_many() instead.
    """

    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The directory on your computer to upload. Files in the directory and its
    # subdirectories will be uploaded. An empty string means "the current
    # working directory".
    # source_directory=""

    # The maximum number of processes to use for the operation. The performance
    # impact of this value depends on the use case, but smaller files usually
    # benefit from a higher number of processes. Each additional process occupies
    # some CPU and memory resources until finished. Threads can be used instead
    # of processes by passing `worker_type=transfer_manager.THREAD`.
    # workers=8

    from pathlib import Path

    from google.cloud.storage import Client, transfer_manager

    storage_client = Client(project=GCP_PROJECT)
    bucket = storage_client.bucket(BUCKET_NAME)

    # Generate a list of paths (in string form) relative to the `directory`.
    # This can be done in a single list comprehension, but is expanded into
    # multiple lines here for clarity.

    # First, recursively get all files in `directory` as Path objects.
    directory_as_path_obj = Path(source_directory)
    paths = directory_as_path_obj.rglob("*")

    # Filter so the list only includes files, not directories themselves.
    file_paths = [path for path in paths if path.is_file()]

    # These paths are relative to the current working directory. Next, make them
    # relative to `directory`
    relative_paths = [path.relative_to(source_directory) for path in file_paths]

    # Finally, convert them all to strings.
    string_paths = [str(path) for path in relative_paths]

    print("\nFound {} files to upload.".format(len(string_paths)))

    # Start the upload.
    results = transfer_manager.upload_many_from_filenames(
        bucket, string_paths, source_directory=source_directory, max_workers=workers
    )

    for name, result in zip(string_paths, results):
        # The results list is either `None` or an exception for each filename in
        # the input list, in order.

        if isinstance(result, Exception):
            print("Failed to upload {} due to exception: {}".format(name, result))
        else:
            print("Uploaded {} to {}.".format(name, bucket.name))


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

def convert_bloob_name_list(bloob_list, path)-> list:
    """
    Function to convert a blood name list into a path directory

    Input:
        bloob name list
    Output:
        local file path list download from a bucket
    """
    path_list = []

    for elt in bloob_list:
        end_ = str.split(elt, "/")[-3:]
        path_list.append(os.path.join(path, end_[0], end_[1], end_[2]))

    return path_list

def test_nb_mask():
    path = os.path.join(LOCAL_DATA_PATH, "ok", "_preprocessed", "y")

    file_list = ["urbansyn_depth_1026_pre.npy",
                 "urbansyn_depth_4777_pre.npy",
                 "urbansyn_depth_0892_pre.npy",
                 "urbansyn_depth_0014_pre.npy"]

    for f in file_list:
        path_file = os.path.join(path, f)
        npy = np.load(path_file)
        print(f"File: {f}   max= {np.max(npy)}")

if __name__ == '__main__':
    #get_blob_url('urbansyn/rgb/rgb_7539.png')
    #gcp_list_files(prefix='_preprocessed/X',extension=None) #urbansyn/depth
    #download_chunks_concurrently('urbansyn/depth/depth_0001.exr', 'depth_0001.exr', chunk_size=64 * 1024 * 1024, workers=8)
    #download_many_blobs_with_transfer_manager(['make3d/test/depth/depth_sph_corr-op36-p-282t000.mat', 'make3d/test/depth/depth_sph_corr-op36-p-313t000.mat'], destination_directory="", workers=8)
    #upload_many_blobs_with_transfer_manager(['00019_00183_indoors_000_010_depth.npy','00019_00183_indoors_000_010_depth_mask.npy'], source_directory="/home/jbo/code/soapoperator/depth-planes-from-2d/raw_data/val/indoors/scene_00019/scan_00183", workers=8)
    # upload_directory_with_transfer_manager(source_directory="/home/pouil/code/soapoperator/depth-planes-from-2d/raw_data/ok/_preprocessed/y", workers=8)
    #clean_data('/home/jbo/code/soapoperator/depth-planes-from-2d/raw_data/tmp/')
    #local_list_files('/home/jbo/code/soapoperator/depth-planes-from-2d/raw_data/tmp')
    #get_npy(f'{LOCAL_DATA_PATH}/ok/_preprocessed/X')
    #test_nb_mask()
    pass
