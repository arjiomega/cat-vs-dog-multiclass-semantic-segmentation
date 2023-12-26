import shutil
from pathlib import Path

import cv2
import numpy as np


class Save:
    """Save or Transfer a File"""

    def __init__(self, savepath):
        """Initialize Save class with a specified savepath.

        Args:
            savepath (Path): The base directory path for saving files.
        """
        self.savepath = savepath

    def save_array(self, array: np.ndarray, savefilename: str, create_dir: Path|str = None):
        """Save either an image or a mask numpy array.

        Args:
            array (np.ndarray): An image or mask numpy array to be saved.
            create_dir (str, optional): Subdirectory to create within the base savepath. Defaults to None.
        """
        if create_dir:
            savepath = Path(self.savepath, create_dir)
            savepath.mkdir(parents=True, exist_ok=True)

        else:
            savepath = self.savepath

        cv2.imwrite(str(Path(savepath, savefilename)), array)

    def transfer_file(
        self, sourcepath: Path, sourcefilename: str, create_dir: str = None
    ):
        """Transfer a file from a source directory to the savepath.

        Args:
            sourcepath (Path): The source directory path.
            sourcefilename (str): The name of the file to transfer.
            create_dir (str, optional): Subdirectory to create within the base savepath. Defaults to None.
        """
        if create_dir:
            savepath = Path(self.savepath, create_dir)
            savepath.mkdir(parents=True, exist_ok=True)

        else:
            savepath = self.savepath

        shutil.copy(Path(sourcepath, sourcefilename), savepath)
