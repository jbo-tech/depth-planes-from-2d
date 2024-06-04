# Preprocessing
from depth_planes.params import *
import tensorflow as tf

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

def preprocess_exr_to_array(path: str) -> np.ndarray:
    """
    _summary_
    """
    pass

def preprocess_jpg_to_array(path: str) -> np.ndarray:
    """
    _summary_
    """
    pass

def preprocess_png_to_array(path: str) -> np.ndarray:
    """
    _summary_
    """
    pass

def preprocess_mat_to_array(path: str) -> np.ndarray:
    """
    _summary_
    """
    pass
