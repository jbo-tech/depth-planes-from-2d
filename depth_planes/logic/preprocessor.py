# Preprocessing
from depth_planes.params import *
from depth_planes.logic.data import *
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import io, image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import scipy.io
import h5py


def preprocess_dataset():

    if DATA_URBANSYN:
        X_path="urbansyn/rgb"
        y_path="urbansyn/depth"
        X_path_preprocessed
        y_path_preprocessed

        file_list = get_data()
        preprocess_bulk(file_list)

        # Upload tmp files
        #upload_directory_with_transfer_manager(source_directory=str(tmp_folder), workers=8)

    if DATA_MAKE3D:
        pass

    if DATA_DIODE:
        pass

    if DATA_MEGADEPTH:
        pass

    if DATA_DIMLRGBD:
        pass

    if DATA_NYUDEPTHV2:
        pass


def preprocess_bulk(files: list,path: str):
    file_list = get_data(path='make3d/test/depth')
    print(file_list)

    # Parameters
    PREPROCESS_CHUNK_SIZE=2
    tmp_folder = Path(LOCAL_DATA_PATH).joinpath("tmp")
    bucket_size = round(len(file_list) / PREPROCESS_CHUNK_SIZE)
    bucket_size = 4

    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    print(tmp_folder)

    for i in range(bucket_size):

        # Donwload a chunk
        chunk_start = i * PREPROCESS_CHUNK_SIZE
        chunk_end = i * PREPROCESS_CHUNK_SIZE+ PREPROCESS_CHUNK_SIZE + 1 if i < bucket_size else None
        chunk_to_download = file_list[chunk_start:chunk_end]
        download_many_blobs_with_transfer_manager(chunk_to_download, destination_directory=tmp_folder, workers=8)

        # Preprocess local file
        path_preprocessed = []
        files_in_tmp = local_list_files(tmp_folder)
        for f in files_in_tmp:
            preprocess_one_image(f)

        # Clean the tmp folder
        clean_data(Path(LOCAL_DATA_PATH).joinpath('tmp','*'))

    return "Preprocess local: Ok"


def preprocess_one_image(path: str, type: str) -> np.ndarray:
    """
    _summary_

    Args:
        X (pd.DataFrame): _description_

    Returns:
        np.ndarray: _description_
    """

    if type == 'img':
        file_array = preprocess_img_to_array(path)

    if type == 'exr':
        file_array = preprocess_exr_to_array(path, log_scale_near=10, log_scale_far=1, log_scale_medium=5)

    if type == 'mat':
        pass

    if type == 'h5':
        pass

    #return file_array


def preprocess_exr_to_array(path, log_scale_near=10, log_scale_far=1, log_scale_medium=5) -> np.ndarray:
    """
    l'image(y) est chargé depuis son path
    """

    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    img_normalized = img / np.max(img)

    img_log_near = np.log1p(img_normalized * log_scale_near)
    img_log_far = np.log1p(img_normalized * log_scale_far)
    img_log_medium = np.log1p(img_normalized * log_scale_medium)

    img_log_combined = img_log_near * 0.33 + img_log_far * 0.33 + img_log_medium * 0.33

    img_log_combined_scaled = img_log_combined / np.max(img_log_combined) * 65535
    img_log_combined_scaled[img_log_combined_scaled > 65535] = 65535
    png_img = img_log_combined_scaled.astype('uint16')

    res = cv2.resize(png_img, dsize=(512, 256), interpolation=cv2.INTER_CUBIC)

    return res


def preprocess_img_to_array(path: str) -> np.ndarray:
    """
    _summary_
    """
    # print('**************************', path)

    img = load_img(path)
    load_image_to_array = img_to_array(img)
    img_standardization = tf.image.per_image_standardization(load_image_to_array)
    img_x = tf.image.resize(img_standardization, [256, 512], preserve_aspect_ratio=True)

    # print('**********************', img_x)
    return img_x


def preprocess_mat_to_array(path: str) -> np.ndarray:
    """
    _summary_
    """
    mat = scipy.io.loadmat(mat_path)
    return mat

def preprocess_h5_to_array(path: str) -> np.ndarray:
    """
    _summary_
    """
    hf = h5py.File(path, 'r')
    return hf

if __name__ == '__main__':
    # preprocess_img_to_array('../../raw_data/rgb/rgb_0001.png')
    # preprocess_mat_to_array('/home/jbo/code/soapoperator/depth-planes-from-2d/raw_data/make3d/depth/make3d_train_depth_depth_sph_corr-060705-17.10.14-p-018t000.mat')
    # preprocess_exr_to_array('../../raw_data/depth/depth_0005.exr')
    preprocess_bulk('','')
