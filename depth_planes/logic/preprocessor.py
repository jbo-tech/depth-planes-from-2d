# Preprocessing
from params import *
from depth_planes.logic.data import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] ="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
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

import logging

logging.basicConfig(filename=f"{ROOT_DIRECTORY}/depth_planes.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def preprocess_dataset():
    preprocess_folder = Path(LOCAL_DATA_PATH).joinpath("ok","_preprocessed")

    if not os.path.exists(preprocess_folder):
        os.makedirs(preprocess_folder)

    logger.info(f'\n\n🚩 Preprocess : start!')

    start = time.time()

    if eval(DATA_URBANSYN) == True:
        X_path="urbansyn/rgb"
        y_path="urbansyn/depth"

        X_file_list = gcp_list_files(X_path)
        y_file_list = gcp_list_files(y_path)

        preprocess_bulk( X_file_list,str(preprocess_folder) + "/X",'urbansyn')
        preprocess_bulk( y_file_list,str(preprocess_folder) + "/y",'urbansyn')

    if eval(DATA_MAKE3D) == True:
        X_path="make3d/rgb"
        y_path="make3d/depth"

        X_file_list = gcp_list_files(X_path)
        y_file_list = gcp_list_files(y_path)

        preprocess_bulk( X_file_list,str(preprocess_folder) + "/X",'make3d')
        preprocess_bulk( y_file_list,str(preprocess_folder) + "/y",'make3d')

    if eval(DATA_DIODE) == True:
        X_path="diode/rgb"
        y_path="diode/depth"

        X_file_list = gcp_list_files(X_path)
        y_file_list = gcp_list_files(y_path)

        preprocess_bulk( X_file_list,str(preprocess_folder) + "/X",'diode')
        preprocess_bulk( y_file_list,str(preprocess_folder) + "/y",'diode')

    if eval(DATA_MEGADEPTH) == True:
        X_path="megadepth/rgb"
        y_path="megadepth/depth"

        X_file_list = gcp_list_files(X_path)
        y_file_list = gcp_list_files(y_path)

        preprocess_bulk( X_file_list,str(preprocess_folder) + "/X",'megadepth')
        preprocess_bulk( y_file_list,str(preprocess_folder) + "/y",'megadepth')

    if eval(DATA_DIMLRGBD) == True:
        X_path="dimlrgbd/rgb"
        y_path="dimlrgbd/depth"

        X_file_list = gcp_list_files(X_path)
        y_file_list = gcp_list_files(y_path)

        preprocess_bulk( X_file_list,str(preprocess_folder) + "/X",'dimlrgbd')
        preprocess_bulk( y_file_list,str(preprocess_folder) + "/y",'dimlrgbd')

    if eval(DATA_NYUDEPTHV2) == True:
        nyudepthv2_path="nyudepthv2/h5"

        nyudepthv2_file_list = gcp_list_files(nyudepthv2_path)

        preprocess_bulk( nyudepthv2_file_list, str(preprocess_folder),'nyudepthv2')

    # Upload tmp files
    #upload_directory_with_transfer_manager(source_directory=str(os.path.dirname(preprocess_folder)), workers=8)

    end = time.time()
    logger.info(f'\n\n✅ Preprocess : ok! ({time.strftime("%H:%M:%S", time.gmtime(end - start))})\n############################################')
    #print(f'\n✅ Preprocess ({time.strftime("%H:%M:%S", time.gmtime(end - start))})')

def preprocess_bulk(files: list, preprocess_path: str, dataset_prefix: str):
    """
    _summary_

    Args:
        files (list): _description_
        preprocess_path (str): _description_
        dataset_prefix (str): _description_

    Returns:
        _type_: _description_
    """

    #print(files)

    # Parameters
    PREPROCESS_CHUNK_SIZE=200
    tmp_folder = os.path.join(LOCAL_DATA_PATH, "tmp")
    bucket_size = round(len(files) / PREPROCESS_CHUNK_SIZE)
    #bucket_size = 4

    print(f'Files to download : {len(files)} in {bucket_size}')
    logger.info(f'\n\nFiles to download : {len(files)} in {bucket_size}\n############################################')

    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    # print(tmp_folder)

    for i in range(bucket_size):

        # Donwload a chunk
        chunk_start = i * PREPROCESS_CHUNK_SIZE if i > 0 else 1 # Remove the folder from the list
        chunk_end = i * PREPROCESS_CHUNK_SIZE + PREPROCESS_CHUNK_SIZE if i < bucket_size else len(files)
        chunk_to_download = files[chunk_start:chunk_end]
        download_many_blobs_with_transfer_manager(chunk_to_download, destination_directory=tmp_folder, workers=8)

        # Preprocess local file
        files_in_tmp = local_list_files(tmp_folder)
        #print(files_in_tmp)
        for f in files_in_tmp:
            print(f'Preprocessing : {f}')
            preprocess_one_image(f,preprocess_path,dataset_prefix)

        # Clean the tmp folder
        clean_data(tmp_folder)

    return "Preprocess local: Ok"


def preprocess_one_image(path_original: str, path_destination: str, dataset_prefix: str) -> np.ndarray:
    """
    _summary_

    Args:
        X (pd.DataFrame): _description_

    Returns:
        np.ndarray: _description_
    """

    if not os.path.exists(path_destination):
        os.makedirs(path_destination)

    #print(path_destination)

    path_ext = path_original.split('.')[-1]
    #print(path_ext)
    name = dataset_prefix + "_" + path_original.split('/')[-1].split('.')[-2]+'_pre'
    #path_pre = 'raw_data/'+path.split('/')[-2]


    if path_ext == 'exr':
        pre = preprocess_exr_to_array(path_original) # Return np.array
        return local_save_data(pre, path=path_destination, name=name)
    elif path_ext =='h5':
        rgb_res, depth_res = preprocess_h5_to_array(path_original)
        rgb_path = local_save_data(rgb_res, name=name+'_rgb', path=f"{path_destination}/X")
        depth_path = local_save_data(depth_res, name=name+'_depth', path=f"{path_destination}/y")
        return rgb_path, depth_path
    else:
        pre = preprocess_img_to_array(path_original)
        return local_save_data(pre, path=path_destination, name=name)



def preprocess_exr_to_array(path, log_scale_near=10, log_scale_far=1, log_scale_medium=5) -> np.ndarray:
    """
    l'image(y) est chargé depuis son path
    """

    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)


    img_log_near = np.log1p(img * log_scale_near)
    img_log_far = np.log1p(img * log_scale_far)
    img_log_medium = np.log1p(img * log_scale_medium)

    img_log_combined = img_log_near * 0.33 + img_log_far * 0.33 + img_log_medium * 0.33

    img_log_combined_scaled = img_log_combined / np.max(img_log_combined) * 65535
    img_log_combined_scaled[img_log_combined_scaled > 65535] = 65535
    png_img = img_log_combined_scaled.astype('uint16')

    res = cv2.resize(png_img, dsize=((eval(IMAGE_SHAPE)[1]), (eval(IMAGE_SHAPE)[0])), interpolation=cv2.INTER_CUBIC)
    res = np.expand_dims(res, axis=-1)
    # print(res.shape)
    return res


def preprocess_img_to_array(path: str) -> np.ndarray:
    """
    _summary_
    """

    img_norm_array = img_to_array(load_img(path))/255
    img_res = tf.image.resize(img_norm_array, [(eval(IMAGE_SHAPE)[0]), (eval(IMAGE_SHAPE)[1])], preserve_aspect_ratio=True)
    return img_res


# def preprocess_mat_to_array(path: str) -> np.ndarray:
#     """
#     _summary_
#     """
#     mat = scipy.io.loadmat(mat_path)
#     return mat

def preprocess_h5_to_array(path: str, log_scale_near=10, log_scale_far=1, log_scale_medium=5) -> np.ndarray:
    """
    _summary_
    """
    h5 = h5py.File(path, 'r')

    h5_rgb = h5['rgb']
    h5_rgb_r = (np.moveaxis(h5_rgb, 0, -1))/255
    h5_rgb_res = tf.image.resize(h5_rgb_r, [(eval(IMAGE_SHAPE)[0]), (eval(IMAGE_SHAPE)[1])], preserve_aspect_ratio=False)

    h5_depth = h5['depth'][:]
    img_log_near = np.log1p(h5_depth * log_scale_near)
    img_log_far = np.log1p(h5_depth * log_scale_far)
    img_log_medium = np.log1p(h5_depth * log_scale_medium)
    img_log_combined = img_log_near * 0.33 + img_log_far * 0.33 + img_log_medium * 0.33
    res = cv2.resize(img_log_combined, dsize=[(eval(IMAGE_SHAPE)[1]), (eval(IMAGE_SHAPE)[0])], interpolation=cv2.INTER_CUBIC)
    h5_depth_res = np.expand_dims(res, axis=-1)

    return h5_rgb_res, h5_depth_res








#if __name__ == '__main__':

    #preprocess_dataset()
    # X, y = preprocess_bulk()
    # preprocess_one_image('/home/mathieu/code/MathieuAmacher/depth-planes-from-2d/raw_data/h5/nyudepthv2_train_study_0008_00001.h5', '/home/mathieu/code/MathieuAmacher/depth-planes-from-2d/raw_data/depth', 'test' )
    # preprocess_img_to_array('/home/mathieu/code/MathieuAmacher/depth-planes-from-2d/raw_data/rgb/rgb_0001.png')
    # preprocess_mat_to_array('/home/jbo/code/soapoperator/depth-planes-from-2d/raw_data/make3d/depth/make3d_train_depth_depth_sph_corr-060705-17.10.14-p-018t000.mat')
    # preprocess_exr_to_array('/home/mathieu/code/MathieuAmacher/depth-planes-from-2d/raw_data/depth/depth_0001.exr')
    # preprocess_h5_to_array('/home/mathieu/code/MathieuAmacher/depth-planes-from-2d/raw_data/h5/nyudepthv2_train_study_0008_00001.h5')
