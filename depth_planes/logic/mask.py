import numpy as np
from params import *
from depth_planes.logic.preprocessor import *


def number_mask(y_pred) -> int:
    """
    Function to determine the number of mask to create from an image y_pred
    """

    pass

def create_mask(y_pred, nb_mask: int =None) -> np.array:
    """
    Function to create nb_mask new images from an image predicted y_pred

    Return:
    np.array with 0 and 1 to have different mask
    with the following shape (image_size(heigth, width), nb_mask)
    """

    if nb_mask == None:
        nb_mask = number_mask(y_pred=y_pred)

    return

def create_mask_in_one(y_pred, nb_mask: int =None) -> np.array:
    """
    Function to create an array representing categorical mask

    Input:
    y_pred.shape -> (128,256,1)
    """

    slice_ = (np.max(y_pred)+1)/nb_mask
    img_dim = y_pred #reduce the dim
    mask = np.full((nb_mask,img_dim.shape[0],img_dim.shape[1]),-1)

    for i in np.arange(1, nb_mask+1):
        mask[i-1] = np.where((img_dim>=(i-1)*slice_) & (img_dim<i*slice_), i, 0)

    mask = mask.sum(axis=0)

    return mask

def all_mask_one_image(x_path: str, y_path: str):

    x = preprocess_img_to_array(x_path)
    x = np.dstack((x, np.ones((eval(IMAGE_SHAPE)[0], eval(IMAGE_SHAPE)[1]), 255)))

    y = np.squeeze(np.load(y_path), axis=2)
    mask_array = np.array([])

    for i in np.unique(y):
        mask = (y == i).astype(int)
        mask_x = x * np.expand_dims(mask, axis=-1)
        mask_array = np.append(mask_array, mask_x)

    mask_array = np.reshape(mask_array, (len(np.unique(y)), (eval(IMAGE_SHAPE)[0]), (eval(IMAGE_SHAPE)[1]), 4))

    mask_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(x_path)))), 'mask')
    os.makedirs(mask_dir, exist_ok=True)

    base_name = os.path.basename(x_path).split('.')[0] + '_mask'
    file_path = os.path.join(mask_dir, base_name)

    np.save(file_path, mask_array)

    return file_path



if __name__ == '__main__':
    all_mask_one_image('/home/mathieu/code/MathieuAmacher/depth-planes-from-2d/raw_data/urban/X100/rgb_0001.png', '/home/mathieu/code/MathieuAmacher/depth-planes-from-2d/raw_data/urban/y100_prec/dataset_depth_0001_pre.npy')
