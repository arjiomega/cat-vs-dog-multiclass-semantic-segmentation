import shutil
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from config import config
from src.data.data_utils import preprocess
from src.data.generate import labelstudio_loader

class LoadBaseData:
    def __init__(self,base_data_label_path,random_seed=5):
        random.seed(random_seed)
        self.base_data_label_path = base_data_label_path

    def load_dataframe(self):
        df = pd.read_csv(self.base_data_label_path, comment='#',header=None, sep=" ")
        df.columns = ['image_name','class_index','specie_index','breed_index']
        df = df.sort_values(by = 'image_name', ascending=True)
        df = df.reset_index(drop=True)
        return df
    
    def get_breed(self,image_name) -> str:
        breed_split = image_name.split('_')[:-1]
        breed = "_".join(breed_split)
        return breed
    
    def gen_breed_name(self,df:pd.DataFrame):
        # Generate Cat breed names
        cats_df = df[df.specie_index == 1]
        cats_breed_df = cats_df.groupby('breed_index')['image_name'].first().reset_index()
        cats_breed_df.image_name = cats_breed_df.image_name.apply(lambda image_name: self.get_breed(image_name))
        cats_breed_df.rename(columns={'image_name':'breed_name'}, inplace=True)
        
         # Generate Dog breed names
        dogs_df = df[df.specie_index == 2]
        dogs_breed_df = dogs_df.groupby('breed_index')['image_name'].first().reset_index()
        dogs_breed_df.image_name = dogs_breed_df.image_name.apply(lambda image_name: self.get_breed(image_name))
        dogs_breed_df.rename(columns={'image_name':'breed_name'}, inplace=True)
    
        # update respective class dataframe with new column
        cats_df = pd.merge(cats_df,cats_breed_df,on='breed_index',how='left')
        dogs_df = pd.merge(dogs_df,dogs_breed_df,on='breed_index',how='left')
    
        # Since some of the cats and dogs breed indices are the same, they were processed separately
        concat_df = pd.concat([cats_df,dogs_df]).reset_index()
    
        # Update specie index from 1 and 2 to cat and dog
        concat_df.specie_index = concat_df.specie_index.apply(lambda x: 'cat' if x == 1 else 'dog')
        concat_df.rename(columns={'specie_index':'specie'},inplace=True)
    
        # Only get relevant columns
        output_df = concat_df[['image_name','specie','breed_name']]
    
        return output_df
    
    def data_splitter(self,class_list:list,
                      train_split:float=0.7,
                      valid_split:float=0.2,
                      test_split:float=0.1) -> tuple[list,list,list]:

            assert(round(train_split+valid_split+test_split,2) == 1.0), "data split must equal to 1"
            
            # First split
            split_index = int((1.0 - test_split) * len(class_list))
            
            train_valid_set = class_list[:split_index]
            test_set = class_list[split_index:]
            
            # Second split
            split_index = int((1.0 - valid_split) * len(train_valid_set))
            
            train_set = train_valid_set[:split_index]
            valid_set = train_valid_set[split_index:]
            
            return train_set,valid_set,test_set
    
    def load_base_data(self,df:pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        # separate by class
        cats = df[df.specie == 'cat'].image_name.to_list()
        dogs = df[df.specie == 'dog'].image_name.to_list()
        
        # shuffle data
        random.shuffle(cats)
        random.shuffle(dogs)
        
        # Cat
        train_cat, valid_cat, test_cat = self.data_splitter(cats)
        # Dog
        train_dog, valid_dog, test_dog = self.data_splitter(dogs)
        
        train_set = train_cat + train_dog
        valid_set = valid_cat + valid_dog
        test_set = test_cat + test_dog

        train_set = df[df.image_name.isin(train_set)].reset_index(drop=True)
        valid_set = df[df.image_name.isin(valid_set)].reset_index(drop=True)
        test_set = df[df.image_name.isin(test_set)].reset_index(drop=True)

        return train_set, valid_set, test_set

    def gen_debugging_set(self,train_df:pd.DataFrame,sample_per_breed:int) -> tuple[pd.DataFrame,pd.DataFrame]:
        
        breed_list = train_df.breed_name.unique()
    
        sampling_df = pd.DataFrame()

        for breed in breed_list:
            sampling_df = pd.concat([sampling_df,train_df[train_df.breed_name == breed].sample(n=sample_per_breed)])

        debugging_df = sampling_df.reset_index(drop=True)
        
        # remove samples used for debugging set from training set
        updated_train_df = (train_df[train_df.merge(debugging_df, indicator=True, how='left')['_merge'].eq('left_only')].reset_index(drop=True))
        
        return updated_train_df,debugging_df
    

class MaskLoader:
    def __init__(self):
        pass

    def update_mask(self,mask):
        mask = np.where(mask > 0, 1,0)
        return mask

    def mask_stacker(self,
                     masks_dict:dict[str:list[int]],
                     mask_dimension:tuple[int,int] # height, width
                     ) -> np.ndarray:
        mask_per_class = []
        classes = ["background", "cat", "dog"]
        
        for class_ in classes:
            
            if class_ in masks_dict:
                current_rle = masks_dict[class_]
                temp_mask = labelstudio_loader.rle_to_mask(current_rle,*mask_dimension)
                temp_mask = self.update_mask(temp_mask)
            else:
                # generate a blank mask as a placeholder
                temp_mask = np.zeros(mask_dimension)
                
            mask_per_class.append(temp_mask)
        
        # Stack multiple 2D arrays to get shape of (height,width,classes)
        stacked_mask = np.stack(mask_per_class,axis=-1)
        
        # Turns (height,width,classes) to (height,width) with unique values of different classes
        # instead of [0,1] we get [0,1,2] -> [background,cat,dog]
        flatten_mask = np.argmax(stacked_mask,axis=-1)
                
        return flatten_mask
    
    def mask_saver(self,mask,file_dir,file_name,format='.png'):
        """_summary_

        Args:
            mask (_type_): _description_
            file_dir (_type_): _description_
            file_name (_type_): _description_
            format (str, optional): _description_. Defaults to '.png'.
        """
        file = Path(file_dir,file_name+format)
        cv2.imwrite(str(file),mask)

class GenCustomData:
    def __init__(self,json_path:str):
        self.df = pd.read_json(json_path)
        self.preprocess_df()
        
        self.maskloader = MaskLoader()
        
    def gen_image_filename(self,row) -> str:
        image_filename = row.split("-")[-1]
        
        return image_filename
    
    def gen_rle(self,row) -> dict[str:list[int]]:
        # {cat: rle, dog: rle}
        rle_dict = {}
        for class_ in row[0]['result']:
            rle = class_['value']['rle']
            label = class_['value']['brushlabels'][0]
            
            rle_dict[label] = rle
        
        return rle_dict
      
    def gen_dimension(self,row) -> tuple[int,int]:
        # get dimensions from any class
        width = row[0]['result'][0]['original_width']
        height = row[0]['result'][0]['original_height']
        return (height,width)
        
    def preprocess_df(self):
        self.df["image_filename"] = self.df.file_upload.apply(lambda row: self.gen_image_filename(row))
        self.df["rle"] = self.df.annotations.apply(lambda row: self.gen_rle(row))
        self.df["dimensions"] = self.df.annotations.apply(lambda row: self.gen_dimension(row))

    def save(self,save_path:str):
        for i,row in self.df.iterrows():
            image_filename = row['image_filename']
            mask_filename = row['image_filename'].split(".")[0] + ".png"
            rle = row['rle']
            dimensions = row['dimensions']
            
            mask = self.maskloader.mask_stacker(masks_dict=rle,
                                                mask_dimension=dimensions)
            
            if "cat" and "dog" in rle:
                path_add = "cats_and_dogs"
            elif "cat" in rle:
                path_add = "cats"
            elif "dog" in rle:
                path_add = "dogs"
            else:
                raise KeyError("no classes found.")
                
            # save mask to new directory
            mask_save_path = Path(save_path,path_add,"masks") 
            mask_save_path.mkdir(parents=True,exist_ok=True)
            
            self.maskloader.mask_saver(mask=mask,
                                       file_dir=mask_save_path,
                                       file_name=mask_filename)
            
            # save image to new directory
            image_save_path = Path(save_path,path_add,"images")
            image_save_path.mkdir(parents=True,exist_ok=True)
            
            source_img = Path(config.NEW_DATA_DIR,image_filename)
            
            shutil.copy(src=source_img,dst=image_save_path)
            

if __name__ == '__main__':    
    # load base data
    file_dir = config.RAW_DATA_DIR
    file_name = Path("annotations","list.txt")

    base_loader = LoadBaseData(base_data_label_path=Path(file_dir,file_name))
    df = base_loader.load_dataframe()
    df = base_loader.gen_breed_name(df)
    train_set, valid_set, test_set = base_loader.load_base_data(df)
    train_set, debugging_set = base_loader.gen_debugging_set(train_set,sample_per_breed=1)
    
    # preprocess and save data
    run = preprocess.Run(train_set, valid_set, test_set, debugging_set,show_logs=True)

    run.set_save_path(train_path=Path(config.PROCESSED_DATA_DIR,"train_set"),
                    valid_path=Path(config.PROCESSED_DATA_DIR,"valid_set"),
                    test_path=Path(config.PROCESSED_DATA_DIR,"test_set"),
                    debug_path=Path(config.PROCESSED_DATA_DIR,"debugging_set"))
    run.gen_preprocessed_data()
    
    # load data from label data
    annotation_path = str(Path(config.NEW_DATA_DIR,"new_data_annotations.json"))
    gen_custom_data = GenCustomData(json_path=annotation_path)
    save_path = Path(config.PROCESSED_DATA_DIR,"debugging_set")
    gen_custom_data.save(save_path=save_path)