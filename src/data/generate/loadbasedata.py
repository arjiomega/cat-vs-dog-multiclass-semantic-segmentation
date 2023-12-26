import random
from pathlib import Path

import pandas as pd

from config import config
from src.data.load import load
from src.data.save import save
from src.data.preprocess import preprocess


class LoadBaseData:
    def __init__(
        self, labelpath, labelfilename, show_logs: bool = False, random_seed=5
    ):
        random.seed(random_seed)
        self.show_logs = show_logs
        self.fullpath = Path(labelpath, labelfilename)

    def load_dataframe(self):
        df = pd.read_csv(self.fullpath, comment="#", header=None, sep=" ")
        df.columns = ["image_name", "class_index", "specie_index", "breed_index"]
        df = df.sort_values(by="image_name", ascending=True)
        df = df.reset_index(drop=True)

        df.rename(columns={"specie_index": "specie"}, inplace=True)
        df.specie = df.specie.apply(lambda row: "cat" if row == 1 else "dog")

        df.drop(["breed_index"], inplace=True, axis="columns")

        return df

    def data_splitter(
        self,
        class_list: list,
        train_split: float = 0.7,
        valid_split: float = 0.2,
        test_split: float = 0.1,
    ) -> tuple[list, list, list]:
        assert (
            round(train_split + valid_split + test_split, 2) == 1.0
        ), "data split must equal to 1"

        # First split
        split_index = int((1.0 - test_split) * len(class_list))

        train_valid_set = class_list[:split_index]
        test_set = class_list[split_index:]

        # Second split
        split_index = int((1.0 - valid_split) * len(train_valid_set))

        train_set = train_valid_set[:split_index]
        valid_set = train_valid_set[split_index:]

        return train_set, valid_set, test_set

    def load_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = self.load_dataframe()

        # separate by class
        cats = df[df.specie == "cat"].image_name.to_list()
        dogs = df[df.specie == "dog"].image_name.to_list()

        # shuffle data
        random.shuffle(cats)
        random.shuffle(dogs)

        # split class
        train_cat, valid_cat, test_cat = self.data_splitter(cats)
        train_dog, valid_dog, test_dog = self.data_splitter(dogs)

        # combine classes
        train_set = train_cat + train_dog
        valid_set = valid_cat + valid_dog
        test_set = test_cat + test_dog

        train_set = df[df.image_name.isin(train_set)].reset_index(drop=True)
        valid_set = df[df.image_name.isin(valid_set)].reset_index(drop=True)
        test_set = df[df.image_name.isin(test_set)].reset_index(drop=True)

        return train_set, valid_set, test_set

    def save_basedata(
        self,
        train_set: pd.DataFrame,
        valid_set: pd.DataFrame,
        test_set: pd.DataFrame,
        savepath: Path,
    ):
        datasets_list = [train_set, valid_set, test_set]
        datasets_savepath_list = [
            Path(savepath, dataset)
            for dataset in ["train_set", "valid_set", "test_set"]
        ]

        saver = save.Save(savepath=savepath)

        for i, (set_, path_) in enumerate(zip(datasets_list, datasets_savepath_list)):
            for j, image_name in enumerate(set_.image_name):
                # cat or dog
                specie = set_[set_.image_name == image_name].specie.values[0]

                img_filename = f"{image_name}.jpg"
                mask_filename = f"{image_name}.png"

                loaded_mask = load.load_mask(
                    filepath=config.RAW_MASK_DIR, filename=mask_filename
                )
                fixed_mask = preprocess.update_mask_values(
                    mask=loaded_mask, specie=specie
                )

                saver.save_array(
                    array=fixed_mask,
                    savefilename=mask_filename,
                    create_dir=Path(path_, "masks"),
                )

                saver.transfer_file(
                    sourcepath=config.RAW_IMG_DIR,
                    sourcefilename=img_filename,
                    create_dir=Path(path_, "images"),
                )

                if self.show_logs:
                    output_placeholder = ["train", "valid", "test"]
                    if j % 20 == 0:
                        print(
                            f"saving data from {output_placeholder[i]}: {j+1} out of {set_.image_name.count()}"
                        )
