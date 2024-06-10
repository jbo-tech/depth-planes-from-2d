import numpy as np


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
    img_dim = np.squeeze(y_pred, axis=-1) #reduce the dim
    mask = np.full((nb_mask,img_dim.shape[0],img_dim.shape[1]),-1)

    for i in np.arange(1, nb_mask+1):
        mask[i-1] = np.where((img_dim>=(i-1)*slice_) & (img_dim<i*slice_), i, 0)

    mask = mask.sum(axis=0)

    return mask

def apply_mask(x, mask:np.array) -> np.array:
    """
    Function to apply mask on the reel image X
    Save the diffrent mask image on a folder

    Return:
    np.array with the following shape (nb_mask, image_size(heigth, width, color))
    """

    pass

def save_mask(mask):
    """
    Function to save the global mask array containning all the masks
    in a temp folder
    """

    pass
