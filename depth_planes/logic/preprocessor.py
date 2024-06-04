# Preprocessing
# from depth_planes.params import *
import tensorflow as tf
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import io, image
from tensorflow.keras.preprocessing.image import img_to_array, load_img

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def preprocess_bulk():
    pass

    # img_list_normalized = []
    # for i in range(len(img_list)):
    #     img_normalize = img_list[i] / 255
    #     img_list_normalized.append(img_normalize)
    #     X = np.array(img_list_normalized)

def preprocess_one_image(paths_list: list, type: str) -> np.ndarray:
    """
    _summary_

    Args:
        X (pd.DataFrame): _description_

    Returns:
        np.ndarray: _description_
    """

    for path in paths_list:

        if type == 'jpg':
            pass

        if type == 'png':
            pass

        if type == 'exr':
            pass

        if type == 'mat':
            pass

        save_data(file,'preprocessed')



# preproc du y
def preprocess_exr_to_array(path: str) -> np.ndarray:
    """
    l'image(y) est chargé depuis son path
    """
    # print('le path***********************************************', path)

    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    im=img*65535
    im[im>65535]=65535
    png_img = im.astype('uint16')

    img_y = cv2.resize(png_img, dsize=(256, 128), interpolation=cv2.INTER_CUBIC)

    # print('*****************', img_y)

    return img_y


#preprocess le X
def preprocess_img_to_array(path: str) -> np.ndarray:
    """
    _summary_
    """
    # print('**************************', path)

    img = load_img(path)
    load_image_to_array = img_to_array(img)
    img_standardization = tf.image.per_image_standardization(load_image_to_array)
    img_x = tf.image.resize(img_standardization, [128, 512], preserve_aspect_ratio=True)

    # print('**********************', img_x)
    return img_x


def preprocess_mat_to_array(path: str) -> np.ndarray:
    """
    _summary_
    """
    pass


if __name__ == '__main__':
    # preprocess_img_to_array('../../raw_data/rgb/rgb_0001.png')
    # preprocess_mat_to_array()
    # preprocess_exr_to_array('../../raw_data/depth/depth_0005.exr')
