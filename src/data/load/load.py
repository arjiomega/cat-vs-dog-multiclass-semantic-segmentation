from pathlib import Path

import cv2
import numpy as np


from .dataloader import DataLoader
from .datasplitter import DataSplitter
from .batchloader import BatchLoader
from src.data.preprocess import preprocess


def load_image(
    filepath: Path|str,
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


def load_batches(data_list, mask_label, img_dir, mask_dir, args):
    """
    Load Batches from Data for Training, Validation, and Testing

    Args:
        data_list (list): List of data.
        mask_label (str): Mask label.
        img_dir (str): Directory containing images.
        mask_dir (str): Directory containing masks.
        args (dict): Arguments containing setup, parameters, and tags:
            - setup (dict): Setup parameters including train, validation, and test splits:
                - train_split (float): Split ratio for training data.
                - valid_split (float): Split ratio for validation data.
                - test_split (float): Split ratio for testing data.
                - random_seed (int): Seed value for randomization.
            - params (dict): Model parameters:
                - batch_size (int): Size of the batches.

    Returns:
        tuple: A tuple containing three BatchLoader objects:
            - batch_train_dataset: BatchLoader for the training dataset.
            - batch_valid_dataset: BatchLoader for the validation dataset.
            - batch_test_dataset: BatchLoader for the testing dataset.

    This function loads and splits data into training, validation, and testing batches. It uses the provided data list, image and mask directories, along with specified splits and batch sizes to create DataLoader and BatchLoader objects for each dataset.

    Note:
        The 'classes' parameter in DataLoader instantiation considers the classes available in the dataset.
    """

    datasplitter = DataSplitter(data_list, mask_label, ".jpg", ".png", args["setup"])

    dataset = datasplitter.split()

    train_set, valid_set, test_set = dataset["train"], dataset["valid"], dataset["test"]

    train_dataset = DataLoader(
        img_dir=img_dir,
        mask_dir=mask_dir,
        img_list=train_set["img"],
        mask_list=train_set["mask"],
        classes=["background", "cat", "dog"],
    )  # consider removing this

    valid_dataset = DataLoader(
        img_dir=img_dir,
        mask_dir=mask_dir,
        img_list=valid_set["img"],
        mask_list=valid_set["mask"],
        classes=["background", "cat", "dog"],
    )  # consider removing this

    test_dataset = DataLoader(
        img_dir=img_dir,
        mask_dir=mask_dir,
        img_list=test_set["img"],
        mask_list=test_set["mask"],
        classes=["background", "cat", "dog"],
    )  # consider removing this

    batch_train_dataset = BatchLoader(
        dataset=train_dataset, batch_size=args["params"]["batch_size"], shuffle=True
    )

    batch_valid_dataset = BatchLoader(
        dataset=valid_dataset, batch_size=args["params"]["batch_size"], shuffle=True
    )

    batch_test_dataset = BatchLoader(
        dataset=test_dataset, batch_size=args["params"]["batch_size"], shuffle=True
    )

    return batch_train_dataset, batch_valid_dataset, batch_test_dataset
