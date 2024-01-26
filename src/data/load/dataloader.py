from pathlib import Path

import cv2
import numpy as np


class DataLoader:
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        img_list: list[str],
        mask_list: list[str],
        n_classes: int,
    ):
        """
        Class for loading image and mask data for semantic segmentation.

        Args:
        - img_dir (str): Directory path containing image files.
        - mask_dir (str): Directory path containing mask files.
        - img_list (list): List of image IDs.
        - mask_list (list): List of mask IDs corresponding to image IDs.
        - n_classes (list): Number of classes.

        Attributes:
        - img_dir (str): Directory path containing image files.
        - mask_dir (str): Directory path containing mask files.
        - img_ids (list): List of image IDs.
        - mask_ids (list): List of mask IDs corresponding to image IDs.
        - img_path (list): List of image file paths.
        - mask_path (list): List of mask file paths.

        Methods:
        - __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray]: Method to retrieve image and mask pair.
        - img_preprocess(self, img: np.ndarray) -> np.ndarray: Method to preprocess images.
        - mask_to_onehot(self, mask: np.ndarray) -> np.ndarray: Method to convert mask to one-hot encoding.
        """

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ids = img_list
        self.mask_ids = mask_list
        self.n_classes = n_classes

        self.img_path = [str(Path(self.img_dir, img_id)) for img_id in self.img_ids]
        self.mask_path = [
            str(Path(self.mask_dir, mask_id)) for mask_id in self.mask_ids
        ]

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieves an image and its corresponding mask.

        Args:
        - i (int): Index to retrieve the image and mask.

        Returns:
        - Tuple[np.ndarray, np.ndarray]: Tuple containing the preprocessed image and its one-hot encoded mask.
        """

        # read img and mask
        img = cv2.imread(self.img_path[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.img_preprocess(img)

        mask = cv2.imread(self.mask_path[i], 0)
        mask = self.mask_to_onehot(mask)

        return img, mask

    def img_preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocesses the input image.

        Args:
        - img (np.ndarray): Input image.

        Returns:
        - np.ndarray: Preprocessed image.
        """

        # resize
        preprocess_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        # normalize
        preprocess_img = (preprocess_img / 127.5) - 1.0

        return preprocess_img

    def mask_to_onehot(self, mask: np.ndarray) -> np.ndarray:
        """
        Converts mask to one-hot encoding.

        Args:
        - mask (np.ndarray): Input mask.

        Returns:
        - np.ndarray: One-hot encoded mask.
        """

        # separate classes into their own channels
        ## if mask values in list of classes (class_values) return true
        masks = [(mask == class_) for class_ in range(self.n_classes)]

        ## stack 2d masks (list) into a single 3d array (True = 1, False = 0)
        ## axis = -1 > add another axis || example shape (500,500, n_classes)
        mask = np.stack(masks, axis=-1).astype("float")

        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_LINEAR)
        onehot_mask = np.round(mask)

        return onehot_mask
