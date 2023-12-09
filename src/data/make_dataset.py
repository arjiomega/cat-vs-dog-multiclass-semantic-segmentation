from pathlib import Path

from config import config
from src.data.data_utils import load_list, load_label_dict, remove_unlabeled, preprocess

if __name__ == '__main__':

    data_list = load_list.load(config.RAW_IMG_DIR,config.RAW_MASK_DIR)
    img_list = [f"{img}.jpg" for img in data_list]
    mask_list = [f"{mask}.png" for mask in data_list]
    specie_dict = load_label_dict.load(config.RAW_DATA_DIR,Path("annotations","list.txt"))
    img_list, mask_list = remove_unlabeled.remove(img_list,mask_list,specie_dict) # type: ignore

    # saving
    run = preprocess.Run(img_list=img_list,
                   mask_list=mask_list,
                   specie_dict=specie_dict)
    run.save_preprocessed_data()