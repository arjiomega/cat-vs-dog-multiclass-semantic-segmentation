import os
from config import config

def load(img_dir=config.RAW_IMG_DIR,mask_dir=config.RAW_MASK_DIR):
    img_list = [img_ for img_ in os.listdir(img_dir) if not img_.startswith(".") and (img_.endswith(".jpg") or img_.endswith(".png"))]
    mask_list =[mask_ for mask_ in os.listdir(mask_dir) if not mask_.startswith(".") and (mask_.endswith(".jpg") or mask_.endswith(".png"))]
    img_list = sorted(img_list)
    mask_list = sorted(mask_list)
    
    return img_list, mask_list