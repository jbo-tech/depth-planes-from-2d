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
    preprocess_folder = Path(LOCAL_DATA_PATH).joinpath("ok","_preprocessed")

    if not os.path.exists(preprocess_folder):
        os.makedirs(preprocess_folder)

    if DATA_URBANSYN:
        X_path="urbansyn/rgb"
        y_path="urbansyn/depth"

        X_file_list = gcp_list_files(X_path)
        preprocess_bulk(X_file_list, str(preprocess_folder) + "/X")

        y_file_list = gcp_list_files(y_path)
        preprocess_bulk(y_file_list, str(preprocess_folder) + "/y")

        # Upload tmp files
        upload_directory_with_transfer_manager(source_directory=str(preprocess_folder), workers=8)

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


def preprocess_bulk(files: list, preprocess_path: str):

    #print(files)

    # Parameters
    PREPROCESS_CHUNK_SIZE=2
    tmp_folder = Path(LOCAL_DATA_PATH).joinpath("tmp")
    bucket_size = round(len(files) / PREPROCESS_CHUNK_SIZE)
    print(bucket_size)
    bucket_size = 4

    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    #print(tmp_folder)

    for i in range(bucket_size):

        # Donwload a chunk
        chunk_start = i * PREPROCESS_CHUNK_SIZE
        chunk_end = i * PREPROCESS_CHUNK_SIZE+ PREPROCESS_CHUNK_SIZE + 1 if i < bucket_size else None
        chunk_to_download = files[chunk_start:chunk_end]
        download_many_blobs_with_transfer_manager(chunk_to_download[1:], destination_directory=tmp_folder, workers=8)

        # Preprocess local file
        files_in_tmp = local_list_files(tmp_folder)
        print(files_in_tmp)
        for f in files_in_tmp:
            print(f'Preprocessing : {f}')
            preprocess_one_image(f,preprocess_path)

        # Clean the tmp folder
        clean_data(tmp_folder)

    return "Preprocess local: Ok"


def preprocess_one_image(path_original: str, path_destination: str) -> np.ndarray:
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
    print(path_ext)
    name = path_original.split('/')[-1].split('.')[-2]+'_pre'
    #path_pre = 'raw_data/'+path.split('/')[-2]


    if path_ext == 'exr':
        pre = preprocess_exr_to_array(path_original) # Return np.array
        return save_data(pre, path=path_destination, name=name)
    else:
        pre = preprocess_img_to_array(path_original)
        return save_data(pre, path=path_destination, name=name)



# preproc du y
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

    res = cv2.resize(png_img, dsize=((eval(IMAGE_SHAPE)[1]), (eval(IMAGE_SHAPE)[0])), interpolation=cv2.INTER_CUBIC)

    return res


# img = cv2.imread('../raw_data/depth/depth_0005.exr', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

# a = preprocess_exr_to_array(path, log_scale_near=10, log_scale_far=1, log_scale_medium=5)

# plt.imshow(a, cmap='gnuplot_r')  # Utiliser une colormap inversée pour plus de contraste
# plt.colorbar()
# plt.title('Depth Image with Increased Contrast for Near Objects')
# plt.show()


#preprocess le X
def preprocess_img_to_array(path: str) -> np.ndarray:
    """
    _summary_
    """
    # print('**************************', path)

    img = load_img(path)
    load_image_to_array = img_to_array(img)
    img_standardization = tf.image.per_image_standardization(load_image_to_array)
    img_x = tf.image.resize(img_standardization, [(eval(IMAGE_SHAPE)[0]), (eval(IMAGE_SHAPE)[1])], preserve_aspect_ratio=True)

    # print('**********************', img_x)
    return img_x


# def preprocess_mat_to_array(path: str) -> np.ndarray:
#     """
#     _summary_
#     """
#     mat = scipy.io.loadmat(mat_path)
#     return mat

# def preprocess_h5_to_array(path: str) -> np.ndarray:
#     """
#     _summary_
#     """
#     hf = h5py.File(path, 'r')

if __name__ == '__main__':

    preprocess_dataset()
    # X, y = preprocess_bulk()
    # preprocess_one_image(path)
    # preprocess_img_to_array('../../raw_data/rgb/rgb_0001.png')
    # preprocess_mat_to_array('/home/jbo/code/soapoperator/depth-planes-from-2d/raw_data/make3d/depth/make3d_train_depth_depth_sph_corr-060705-17.10.14-p-018t000.mat')
    # preprocess_exr_to_array('../../raw_data/depth/depth_0005.exr')
