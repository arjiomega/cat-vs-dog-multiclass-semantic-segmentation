import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from config import config

class Preprocess:
    def __init__(self,file,label):
        self.file = file
        self.label = label

    def load(self):
        mask_file = os.path.join(config.RAW_MASK_DIR,self.file)
        mask = cv2.imread(mask_file,0)

        return mask

    def fix(self,mask):
        specie_index = self.label

        bg = 0
        condition_object = np.logical_or(mask==1,mask==3)

        fixed_mask = np.where(condition_object,specie_index,bg)

        classes = ["background","cat","dog"]
        class_val = [i for i,_ in enumerate(classes)]
        mask_per_class = [(fixed_mask == class_) for class_ in class_val]

        # turns list of 2D masks to single 3D mask of shape (width,height,n_classes)
        fixed_mask = np.stack(mask_per_class, axis = -1).astype('float')

        # turns onehot mask where each channel contains 1s and 0s to a single channel 
        # with a value from 0 to number of classes
        fixed_mask = np.argmax(fixed_mask, axis=2)

        return fixed_mask
    
    def save(self):
        mask = self.load()
        mask = self.fix(mask)
        cv2.imwrite(os.path.join(config.PROCESSED_MASK_DIR,self.file),mask)

class Run:
    def __init__(self,img_list,mask_list,specie_dict):
        self.img_list = img_list
        self.mask_list = mask_list
        self.specie_dict = specie_dict

    def remove_ext(self,file_name):
        file_name = file_name.split(".")[0]
        return file_name

    def save_image(self,file):
        source_img = Path(config.RAW_IMG_DIR,file)
        shutil.copy(source_img,config.PROCESSED_IMG_DIR)

    def save_preprocessed_data(self):
        for img, mask in zip(self.img_list,self.mask_list):
             file_name = self.remove_ext(mask)
             preprocess = Preprocess(mask,self.specie_dict[file_name])
             preprocess.save()

             self.save_image(img)


