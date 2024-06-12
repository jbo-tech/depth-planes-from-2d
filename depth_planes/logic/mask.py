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

def create_mask_from_image(x_path :str, y_path: str, y_prec_path):

    x_array = img_to_array(load_img(x_path))
    x_reshape = np.reshape(x_array, (x_array.shape[0]*x_array.shape[1], 3))

    y_array = cv2.imread(y_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    y_reshape = np.reshape(y_array, (y_array.shape[0]*y_array.shape[1]))

    y_mask_image = np.load(y_prec_path)
    y_mask_enlarge = cv2.resize(y_mask_image, dsize=(y_array.shape[1], y_array.shape[0]), interpolation=cv2.INTER_NEAREST)
    y_mask_reshape = np.reshape(y_mask_enlarge, (y_mask_enlarge.shape[0]*y_mask_enlarge.shape[1], 1))


    mask_array = np.array([])

    for i in np.unique(y_mask_reshape):
        mask = (y_mask_reshape == i).astype(int)
        mask_x = x_reshape * mask
        mask_array = np.append(mask_array, mask_x)

    mask_array = np.reshape(mask_array, (int(mask_array.shape[0]/3), 3))

    mask_array = mask_array/255

    mask_a = np.expand_dims((mask_array.sum(axis=1) > 0).astype('float64'), axis=-1)
    mask_b = np.concatenate([mask_array,mask_a], axis=1)

    rgba_array = np.reshape(np.asarray(mask_b), (len(np.unique(y_mask_image)), y_array.shape[0], y_array.shape[1], 4))

    return rgba_array



if __name__ == '__main__':
    all_mask_one_image('/home/mathieu/code/MathieuAmacher/depth-planes-from-2d/raw_data/urban/X100/rgb_0001.png', '/home/mathieu/code/MathieuAmacher/depth-planes-from-2d/raw_data/urban/y100_prec/dataset_depth_0001_pre.npy')
