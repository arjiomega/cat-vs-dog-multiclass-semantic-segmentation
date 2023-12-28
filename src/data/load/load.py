from pathlib import Path

import cv2
import numpy as np

from src.data.preprocess import preprocess


def load_image(
    filepath: Path | str,
    filename: str,
    preprocess_list: list[str] = [],
    shape: tuple[int, int] = None,
    normalize_range: tuple[float, float] = None,
) -> np.ndarray:
    fullpath = Path(filepath, filename)

    image = cv2.imread(str(fullpath))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if "resize" in preprocess_list:
        if not shape:
            raise ValueError("shape is required if resize.")
        else:
            image = preprocess.resize(image, shape)
    if "normalize" in preprocess_list:
        if not normalize_range:
            raise ValueError("normalize_range is required if normalize.")
        else:
            image = preprocess.normalize(image, *normalize_range)

    return image


def load_mask(
    filepath: str,
    filename: str,
    preprocess_list: list[str] = [],
    shape: tuple[int, int] = None,
) -> np.ndarray:
    """Load a mask image from a specified filepath and filename.

    ### Args:
    - filepath (str): The directory path where the mask file is located.
    - filename (str): The name of the mask file to load.

    ### Returns:
    - np.ndarray: The loaded mask as a NumPy array.

    ### Description:
    This function loads a mask image from the provided filepath and filename using OpenCV's
    imread function with 0 as the argument to read the image as grayscale. The loaded mask
    image is returned as a NumPy array.

    ### Note:
    The function assumes that the mask file is in a valid image format readable by OpenCV.
    """

    fullpath = Path(filepath, filename)
    mask = cv2.imread(str(fullpath), 0)

    if "resize" in preprocess_list:
        if not shape:
            raise ValueError("shape is required if resize.")
        else:
            mask = preprocess.resize(mask, shape)

    return mask
