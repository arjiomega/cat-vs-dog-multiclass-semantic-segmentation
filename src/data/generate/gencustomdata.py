import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from config import config
from . import labelstudio_loader
from src.data.save import save

class MaskLoader:
    def __init__(self):
        pass

    def update_mask(self, mask):
        mask = np.where(mask > 0, 1, 0)
        return mask

    def mask_stacker(
        self,
        masks_dict: dict[str : list[int]],
        mask_dimension: tuple[int, int],  # height, width
    ) -> np.ndarray:
        mask_per_class = []
        classes = ["background", "cat", "dog"]

        for class_ in classes:
            if class_ in masks_dict:
                current_rle = masks_dict[class_]
                temp_mask = labelstudio_loader.rle_to_mask(current_rle, *mask_dimension)
                temp_mask = self.update_mask(temp_mask)
            else:
                # generate a blank mask as a placeholder
                temp_mask = np.zeros(mask_dimension)

            mask_per_class.append(temp_mask)

        # Stack multiple 2D arrays to get shape of (height,width,classes)
        stacked_mask = np.stack(mask_per_class, axis=-1)

        # Turns (height,width,classes) to (height,width) with unique values of different classes
        # instead of [0,1] we get [0,1,2] -> [background,cat,dog]
        flatten_mask = np.argmax(stacked_mask, axis=-1)

        return flatten_mask

    def mask_saver(self, mask, file_dir, file_name, format=".png"):
        file = Path(file_dir, file_name + format)
        cv2.imwrite(str(file), mask)


class GenCustomData:
    def __init__(self, json_path: Path, savepath:Path):
        self.df = pd.read_json(str(json_path))
        self.preprocess_df()

        self.maskloader = MaskLoader()
        self.saver = save.Save(savepath=savepath)

    def gen_image_filename(self, row) -> str:
        image_filename = row["image"].split("/")[-1]

        return image_filename

    def gen_condition(self, row) -> str:
        condition = row["image"].split("/")[-2]

        return condition

    def gen_rle(self, row) -> dict[str : list[int]]:
        # {cat: rle, dog: rle}
        rle_dict = {}
        for class_ in row[0]["result"]:
            rle = class_["value"]["rle"]
            label = class_["value"]["brushlabels"][0]

            rle_dict[label] = rle

        return rle_dict

    def gen_dimension(self, row) -> tuple[int, int]:
        # get dimensions from any class
        width = row[0]["result"][0]["original_width"]
        height = row[0]["result"][0]["original_height"]
        return (height, width)

    def preprocess_df(self):
        self.df["image_filename"] = self.df.data.apply(
            lambda row: self.gen_image_filename(row)
        )
        self.df["condition"] = self.df.data.apply(lambda row: self.gen_condition(row))
        self.df["rle"] = self.df.annotations.apply(lambda row: self.gen_rle(row))
        self.df["dimensions"] = self.df.annotations.apply(
            lambda row: self.gen_dimension(row)
        )

    def save(self):
        for i, row in self.df.iterrows():
            image_filename = row["image_filename"]
            mask_filename = row["image_filename"].split(".")[0] + ".png"

            condition = row["condition"]

            rle = row["rle"]
            dimensions = row["dimensions"]

            mask = self.maskloader.mask_stacker(
                masks_dict=rle, mask_dimension=dimensions
            )

            # save mask to new directory
            self.saver.save_array(
                array = mask, 
                savefilename=mask_filename,
                create_dir=Path(condition,"masks")
            )
            
            # save image to new directory
            self.saver.transfer_file(
                sourcepath=Path(config.NEW_DATA_DIR, condition),
                sourcefilename=image_filename,
                create_dir=Path(condition,"images")
            )
