import os
from config import config

def load():
    img_list = [img_ for img_ in os.listdir(config.RAW_IMG_DIR) if not img_.startswith(".") and (img_.endswith(".jpg") or img_.endswith(".png"))]
    mask_list =[mask_ for mask_ in os.listdir(config.RAW_MASK_DIR) if not mask_.startswith(".") and (mask_.endswith(".jpg") or mask_.endswith(".png"))]
    img_list = sorted(img_list)
    mask_list = sorted(mask_list)
    
    return img_list, mask_list