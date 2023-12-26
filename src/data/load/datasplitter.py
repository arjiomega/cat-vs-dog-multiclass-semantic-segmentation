import random


class DataSplitter:
    """
    Splits input and mask data into training, validation, and test sets.

    Args:
    - input_list (list): List of input data filenames.
    - mask_label (dict): Dictionary mapping input data filenames to mask labels.
    - img_ext (str): Image file extension.
    - mask_ext (str): Mask file extension.
    - args (dict): Dictionary containing split configuration parameters:
        - "random_seed" (int): Seed for random shuffling.
        - "train_split" (float): Proportion of data for training set.
        - "valid_split" (float): Proportion of data for validation set.
        - "test_split" (float): Proportion of data for test set.

    Methods:
    - class_splitter(file_ext): Splits data by class based on file extension.
    - train_valid_test_split(img_list, mask_list, shuffle): Splits data into training, validation, and test sets.
    - split(): Executes the data splitting process and returns the split datasets.
    """

    def __init__(
        self,
        input_list: list[str],
        mask_label: dict[str, int],
        img_ext: str,
        mask_ext: str,
        args: dict,
    ):
        self.input_list = input_list
        self.mask_label = mask_label
        self.img_ext = img_ext
        self.mask_ext = mask_ext

        self.random_seed = args["random_seed"]
        self.train_split = args["train_split"]
        self.valid_split = args["valid_split"]
        self.test_split = args["test_split"]

    def class_splitter(self, file_ext: str) -> dict[int, list[str]]:
        """
        Splits data by class based on file extension. (WRONG)

        Args:
        - file_ext (str): File extension used for splitting data.

        Returns:
        - dict: Dictionary containing data split by class.
        """
        class_split = {
            class_: [
                img + file_ext
                for img in self.input_list
                if img in self.mask_label and self.mask_label[img] == class_
            ]
            for class_ in set(self.mask_label.values())
        }

        return class_split

    def train_valid_test_split(self, img_list, mask_list, shuffle):
        """
        Splits data into training, validation, and test sets.

        Args:
        - img_list (list): List of image data filenames.
        - mask_list (list): List of mask data filenames.
        - shuffle (bool): Flag to shuffle the data.

        Returns:
        - dict: Dictionary containing splits for training, validation, and test sets.
        """

        if shuffle:
            random.seed(self.random_seed)

            zipped = list(zip(img_list, mask_list))
            random.shuffle(zipped)
            img_list, mask_list = zip(*zipped)

        assert (
            round(self.train_split + self.valid_split + self.test_split, 1) == 1.0
        ), "sum of train,valid,test must be equal to 1"

        train_count = int(self.train_split * (len(img_list)))
        valid_count = int((self.train_split + self.valid_split) * (len(img_list)))
        test_count = int(
            (self.train_split + self.valid_split + self.test_split) * (len(img_list))
        )

        dataset = {
            "train": {"img": img_list[:train_count], "mask": mask_list[:train_count]},
            "valid": {
                "img": img_list[train_count:valid_count],
                "mask": mask_list[train_count:valid_count],
            },
            "test": {
                "img": img_list[valid_count : test_count + 1],
                "mask": mask_list[valid_count : test_count + 1],
            },
        }

        return dataset

    def split(self) -> dict[str, dict[str, list]]:
        """
        Executes the data splitting process and returns the split datasets.

        Returns:
        - dict: Dictionary containing training, validation, and test datasets.
        """

        train_img, train_mask = [], []
        valid_img, valid_mask = [], []
        test_img, test_mask = [], []

        img_list_by_class = self.class_splitter(self.img_ext)
        mask_list_by_class = self.class_splitter(self.mask_ext)

        # iterate over all classes
        for img, mask in zip(img_list_by_class, mask_list_by_class):
            temp_ds = self.train_valid_test_split(
                img_list_by_class[img], mask_list_by_class[mask], shuffle=True
            )
            train_img.extend(temp_ds["train"]["img"])
            train_mask.extend(temp_ds["train"]["mask"])
            valid_img.extend(temp_ds["valid"]["img"])
            valid_mask.extend(temp_ds["valid"]["mask"])
            test_img.extend(temp_ds["test"]["img"])
            test_mask.extend(temp_ds["test"]["mask"])

        return {
            "train": {"img": train_img, "mask": train_mask},
            "valid": {"img": valid_img, "mask": valid_mask},
            "test": {"img": test_img, "mask": test_mask},
        }
