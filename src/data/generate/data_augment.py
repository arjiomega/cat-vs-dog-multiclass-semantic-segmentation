import os
import shutil
from pathlib import Path

import albumentations as A

from config import config
from src.data.load import load
from src.data.save.save import Save
from src.data.preprocess import preprocess


def iter_data_decorator(func):
    def inner(self, *args, **kwargs):
        img_path_list = sorted(os.listdir(self.img_path))
        mask_path_list = sorted(os.listdir(self.mask_path))

        for img_filename, mask_filename in zip(img_path_list, mask_path_list):
            image = load.load_image(filepath=self.img_path, filename=img_filename)
            mask = load.load_mask(filepath=self.mask_path, filename=mask_filename)
            filename = img_filename.split(".")[0]

            func(self, image, mask, filename, *args, **kwargs)

    return inner


class DataAugmenter:
    def __init__(self) -> None:
        self.img_path = Path(config.TRAIN_SET_PATH, "images")
        self.mask_path = Path(config.TRAIN_SET_PATH, "masks")

        self.augment_list = [
            A.RandomCrop(width=100, height=100),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            # A.GaussNoise(always_apply=True, p=1.0, var_limit=(462.17, 500.0)),
            A.MotionBlur(always_apply=True, p=1.0, blur_limit=(12, 15)),
        ]
        self.saver = Save(savepath=Path(config.PROCESSED_DATA_DIR, "augs"))

    def random_augment(self, image, mask):
        transform = A.Compose(self.augment_list)
        transformed = transform(image=image, mask=mask)
        return transformed["image"], transformed["mask"]

    def generate_each_augment(self):
        pass

    @iter_data_decorator
    def apply_augment(self, image, mask, filename, *args, **kwargs):
        image = preprocess.resize(image, shape=(224, 224))
        mask = preprocess.resize(mask, shape=(224, 224))
        img_augment, mask_augment = self.random_augment(image, mask)
        img_augment = preprocess.resize(img_augment, shape=(224, 224))
        mask_augment = preprocess.resize(mask_augment, shape=(224, 224))
        self.saver.save_array(
            array=img_augment,
            savefilename="aug_" + filename + ".jpg",
            create_dir=Path("images"),
        )
        self.saver.save_array(
            array=mask_augment,
            savefilename="aug_" + filename + ".png",
            create_dir=Path("masks"),
        )


if __name__ == "__main__":
    if os.path.exists(Path(config.PROCESSED_DATA_DIR, "augs")):
        shutil.rmtree(Path(config.PROCESSED_DATA_DIR, "augs"))

    data_augmented = DataAugmenter()
    data_augmented.apply_augment()
