"""Preprocess images and masks"""

import cv2
import numpy as np


def update_mask_values(mask: np.ndarray, specie: str) -> np.ndarray:
    """Updates mask values based on the provided species (cat or dog) in the given mask array.

    ### Args:
    - mask (np.ndarray): The mask array representing pet segmentation.
    - specie (str): The species for which the mask values need to be updated ('cat' or 'dog').

    ### Returns:
    - np.ndarray: The updated mask array after differentiating between cat and dog values.

    ### Description:
    The function takes in a mask array where only 'background', 'border', and 'pet' are represented
    as unique values. It differentiates between cat and dog by assigning values to the respective
    pixels based on the given species ('cat' or 'dog').
    """

    bg = 0
    classes = ["background", "cat", "dog"]

    # generate bool array containing the target with the border
    condition_object = np.logical_or(mask == 1, mask == 3)

    # If a value in condition_object is True, set it to the right specie,
    # else set to bg or 0
    fixed_mask = np.where(condition_object, classes.index(specie), bg)

    # Generate onehot encoding of mask
    mask_per_class = [(fixed_mask == class_) for class_, _ in enumerate(classes)]

    # turns list of 2D masks to single 3D mask of shape (width,height,n_classes)
    fixed_mask = np.stack(mask_per_class, axis=-1).astype("float")

    # Flatten mask again to 2d array for saving
    updated_mask = np.argmax(fixed_mask, axis=-1)

    return updated_mask


def resize(array: np.ndarray, shape: tuple[int, int]):
    array = cv2.resize(array, shape, interpolation=cv2.INTER_LINEAR)
    return array


def normalize(
    array: np.ndarray, normalize_min: int | float, normalized_max: int | float
):
    original_min = 0
    original_max = 255

    numerator = (array - original_min) * (normalized_max - normalize_min)
    denominator = original_max - original_min

    normalized_array = normalize_min + (numerator / denominator)

    return normalized_array


def remove_normalization(array: np.ndarray):
    # remove -1,1 img normalization
    return ((array + 1.0) * 127.5) / 255.0


def mask2onehot():
    pass
