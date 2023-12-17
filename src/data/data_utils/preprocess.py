import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from config import config

class Preprocess:
    def __init__(self,file_dir,file_name):
        self.file_dir = file_dir
        self.file_name = file_name
    
    def load(self):
        mask_file = os.path.join(self.file_dir,self.file_name)
        mask = cv2.imread(mask_file,0)
        
        return mask
    
    def fix_mask(self,mask,specie):
        
        bg = 0
        classes = ["background","cat","dog"]
        
        # generate bool array containing the target with the border
        condition_object = np.logical_or(mask==1, mask==3)

        # If a value in condition_object is True, set it to the right specie, 
        # else set to bg or 0
        fixed_mask = np.where(condition_object,classes.index(specie),bg)

        # Generate onehot encoding of mask
        mask_per_class = [(fixed_mask == class_) for class_,_ in enumerate(classes)]

        # turns list of 2D masks to single 3D mask of shape (width,height,n_classes)
        fixed_mask = np.stack(mask_per_class, axis = -1).astype('float')

        # Flatten mask again to 2d array for saving
        fixed_mask = np.argmax(fixed_mask, axis=-1)

        return fixed_mask

class Run:
    def __init__(self, 
                 train_set:pd.DataFrame, 
                 valid_set:pd.DataFrame,
                 test_set:pd.DataFrame,
                 debug_set:pd.DataFrame,
                 show_logs=False):
        
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.debug_set = debug_set
        
        self.show_logs = show_logs
        
    def set_save_path(self,train_path,valid_path,test_path,debug_path):
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.debug_path = debug_path
        
    def save(self,
             img_filename:str,
             mask_filename:str,
             mask:np.ndarray,
             save_path:str):
        # train/images/ and train/masks
        img_path = Path(save_path,'images')
        mask_path = Path(save_path,'masks')
        
        img_path.mkdir(parents=True,exist_ok=True)
        mask_path.mkdir(parents=True,exist_ok=True)
        
        # copy image to new directory
        source_img = Path(config.RAW_IMG_DIR,img_filename)
        shutil.copy(source_img,img_path)
        
        # save preprocessed mask to new directory
        mask_path_n_filename = str(Path(mask_path,mask_filename))
        cv2.imwrite(mask_path_n_filename,mask)

    def gen_preprocessed_data(self):
        data_df_list = [self.train_set,self.valid_set,self.test_set,self.debug_set]
        paths = [self.train_path,self.valid_path,self.test_path,self.debug_path]
        
        for i,(set_,path_) in enumerate(zip(data_df_list,paths)):
            for j,file_name in enumerate(set_.image_name):
            
                # cat or dog
                specie = set_[set_.image_name == file_name].specie.values[0]
                
                img_filename = f'{file_name}.jpg'
                mask_filename = f'{file_name}.png'
                
                # preprocess mask
                preprocess = Preprocess(file_dir=config.RAW_MASK_DIR,
                                        file_name=mask_filename)
                loaded_mask = preprocess.load()
                fixed_mask = preprocess.fix_mask(mask=loaded_mask,specie=specie)
                
                # save image and mask to the new directory
                # update save path if debugging set
                if i == 3:
                    save_path = Path(path_,f'{specie}s')
                else:
                    save_path = path_
                    
                self.save(img_filename=img_filename,
                        mask_filename=mask_filename,
                        mask=fixed_mask,
                        save_path=save_path)
                
                if self.show_logs:   
                    output_placeholder = ["train","valid","test","debugging"]
                    if j % 20 == 0:
                        print(f"saving data from {output_placeholder[i]}: {j+1} out of {set_.image_name.count()}")


