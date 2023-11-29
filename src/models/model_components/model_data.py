import os
import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from config import config

def load(img_list,mask_list,mask_label):

    train_split = 0.9
    valid_split = 0.08
    test_split = 0.02
    rand_seed = 5
    batch_size = 16

    inp_list = [img.split(".")[0] for img in img_list] # use inp_list instead of img_list and mask_list

    datasplitter = DataSplitter(inp_list, mask_label, ".jpg", ".png", rand_seed, 
                                train_split, valid_split, test_split)

    dataset = datasplitter.split()

    train_set,valid_set,test_set = dataset['train'], dataset['valid'], dataset['test']

    train_dataset = Load_Iter(img_dir=config.PROCESSED_IMG_DIR,
                              mask_dir=config.PROCESSED_MASK_DIR,
                              img_list=train_set['img'],
                              mask_list=train_set['mask'],
                              specie_dict=mask_label, # consider removing this
                              classes=['background','cat','dog'], # consider removing this
                              classes_limit=False) # consider removing this
    
    valid_dataset = Load_Iter(img_dir=config.PROCESSED_IMG_DIR,
                              mask_dir=config.PROCESSED_MASK_DIR,
                              img_list=valid_set['img'],
                              mask_list=valid_set['mask'],
                              specie_dict=mask_label, # consider removing this
                              classes=['background','cat','dog'], # consider removing this
                              classes_limit=False) # consider removing this
    
    test_dataset = Load_Iter(img_dir=config.PROCESSED_IMG_DIR,
                              mask_dir=config.PROCESSED_MASK_DIR,
                              img_list=test_set['img'],
                              mask_list=test_set['mask'],
                              specie_dict=mask_label, # consider removing this
                              classes=['background','cat','dog'], # consider removing this
                              classes_limit=False) # consider removing this
    
    batch_train_dataset = BatchLoader(dataset=train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True)
    
    batch_valid_dataset = BatchLoader(dataset=valid_dataset,
                                      batch_size=batch_size,
                                      shuffle=True)
    
    batch_test_dataset = BatchLoader(dataset=test_dataset,
                                      batch_size=batch_size,
                                      shuffle=True)

    return batch_train_dataset, batch_valid_dataset, batch_test_dataset
    
class DataSplitter:
    def __init__(self,input_list,mask_label,img_ext,mask_ext,rand_seed=5,train=0.9,valid=0.08,test=0.02):
        self.input_list = input_list
        self.mask_label = mask_label
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.rand_seed = rand_seed
        self.train = train
        self.valid = valid
        self.test = test

    def class_splitter(self,file_ext):
        class_split = {class_:[img+file_ext for img in self.input_list if img in self.mask_label and self.mask_label[img]==class_]
                       for class_ in set(self.mask_label.values())}

        return class_split

    def train_valid_test_split(self,img_list,mask_list,shuffle):

        if shuffle:
            random.seed(self.rand_seed)

            zipped = list(zip(img_list,mask_list))
            random.shuffle(zipped)
            img_list, mask_list = zip(*zipped)

        assert round(self.train + self.valid + self.test,1) == 1.0, "sum of train,valid,test must be equal to 1"

        train_count = int(self.train*(len(img_list)))
        valid_count = int((self.train+self.valid)*(len(img_list)))
        test_count =  int((self.train+self.valid+self.test)*(len(img_list)))

        dataset = {"train":     {"img":img_list[:train_count],
                                "mask": mask_list[:train_count]},

                "valid":     {"img":img_list[train_count:valid_count],
                                "mask": mask_list[train_count:valid_count]},

                "test":      {"img":img_list[valid_count:test_count+1],
                                "mask": mask_list[valid_count:test_count+1]}
                }
        
        return dataset

    def split(self):
        train_img, train_mask = [], []
        valid_img, valid_mask = [], []
        test_img, test_mask = [], []

        img_list_by_class = self.class_splitter(self.img_ext)
        mask_list_by_class = self.class_splitter(self.mask_ext)

        # iterate over all classes
        for img,mask in zip(img_list_by_class,mask_list_by_class):

            temp_ds = self.train_valid_test_split(img_list_by_class[img],mask_list_by_class[mask],shuffle=True)
            train_img.extend(temp_ds["train"]["img"])
            train_mask.extend(temp_ds["train"]["mask"])
            valid_img.extend(temp_ds["valid"]["img"])
            valid_mask.extend(temp_ds["valid"]["mask"])
            test_img.extend(temp_ds["test"]["img"])
            test_mask.extend(temp_ds["test"]["mask"])

        return {
            "train": {
                "img": train_img,
                "mask": train_mask
            },
            "valid": {
                "img": valid_img,
                "mask": valid_mask
            },
            "test": {
                "img": test_img,
                "mask": test_mask
            }
        }

class Load_Iter:
    def __init__(self,
                img_dir, mask_dir,
                img_list, mask_list,
                specie_dict,
                classes, classes_limit):

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ids = img_list
        self.mask_ids = mask_list

        self.specie_dict = specie_dict
        self.classes = classes

        self.img_path = [str(Path(self.img_dir,img_id)) for img_id in self.img_ids]
        self.mask_path = [str(Path(self.mask_dir,mask_id)) for mask_id in self.mask_ids]

        # Count classes and their number {"cat": 500, "dog": 1500}
        self.class_count = self.class_counter()

        # classes = 0: background || 1: cat || 2: dog   class_val = [0,1,2]
        if classes_limit:
            self.class_val = [self.classes.index(cls) for cls in classes_limit]
        else:
            self.class_val = [i for i,_ in enumerate(self.classes)]



    def __getitem__(self,i):
        # read img and mask
        img = cv2.imread(self.img_path[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.img_preprocess(img)

        mask = cv2.imread(self.mask_path[i],0)
        mask = self.mask_to_onehot(mask)

        return img, mask

    def img_preprocess(self,img):
        #resize
        preprocess_img = cv2.resize(img, (224,224), interpolation=cv2.INTER_LINEAR)
        #normalize
        preprocess_img = (preprocess_img/127.5) - 1.0

        return preprocess_img

    def mask_to_onehot(self,mask):

        # separate classes into their own channels
        ## if mask values in list of classes (class_values) return true
        masks = [(mask == class_) for class_ in self.class_val]
        ## stack 2d masks (list) into a single 3d array (True = 1, False = 0)
        ## axis = -1 > add another axis || example shape (500,500, n_classes)
        mask = np.stack(masks, axis = -1).astype('float')

        mask= cv2.resize(mask, (224,224), interpolation=cv2.INTER_LINEAR)
        onehot_mask = np.round(mask)

        return onehot_mask

    def class_counter(self):
        # classes=["background","cat","dog"]
        name_list = [class_.split(os.path.sep)[-1].split(".")[0] for class_ in self.img_path]
        specie_list = [self.specie_dict.get(name,None) for name in name_list]
        class_count = {class_name: specie_list.count(class_id) for class_id,class_name in enumerate(self.classes) if class_name != "background"}

        return class_count

class BatchLoader(tf.keras.utils.Sequence):
    def __init__(self,
                 dataset,
                 batch_size,
                 shuffle=False):

        self.dataset = dataset
        self.dataset_size = len(dataset.img_path)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indexes = np.arange(self.dataset_size)

        self.on_epoch_end()

    def __len__(self):
        # Number of Batches per Epoch
        return int(np.floor(self.dataset_size / self.batch_size))  # len(self.list_IDs)

    def __getitem__(self,i):
        start = i * self.batch_size
        stop = (i+1) * self.batch_size
        data = []

        for j in range(start,stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def on_epoch_end(self):
        #self.indexes = np.arange(self.dataset_size)

        if self.shuffle == True:
            np.random.shuffle(self.indexes)