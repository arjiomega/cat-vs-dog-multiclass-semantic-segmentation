from pathlib import Path

import mlflow
import tensorflow as tf

import model_setup
from config import config
from src.data.load import load
from train_model import load_data
from src.models.model_components.callbacks import plot_predict
from src.data.data_utils import load_label_dict, load_list, load_args

if __name__ == "__main__":
    
    args = load_args.load_args(config.CONFIG_DIR,"args.json") 
    
    data_list = load_list.load(config.PROCESSED_IMG_DIR,config.PROCESSED_MASK_DIR)
    mask_label = load_label_dict.load(Path(config.RAW_DATA_DIR,'annotations'),'list.txt') #transfer path of list.txt

    train_set, valid_set, test_set = load.load_batches(data_list,
                                                       mask_label,
                                                       config.PROCESSED_IMG_DIR,
                                                       config.PROCESSED_MASK_DIR,
                                                       args)
    
    
    custom_objects = {metric:model_setup.get_metric(get=metric) for metric in args['setup']['metrics']}
    custom_objects[args['setup']['loss']] = model_setup.get_loss(get=args['setup']['loss'])
    custom_objects["classification_metrics"] = model_setup.get_metric(get="classification_metrics") # required for other metrics
    
    # does not work without compile=False
    model = mlflow.tensorflow.load_model('runs:/2c310585e6494182885b8d5a3c77a316/model')
                                         #saved_model_kwargs=custom_objects)
                                         #keras_model_kwargs={"compile":False})
    
    img,mask = valid_set.dataset[50]
    
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(),
                plot_predict.plot_predict_per_epoch(model,img,mask)
                ]
    
    model_history = model.fit(train_set,
                                  steps_per_epoch = len(train_set),
                                  epochs=args['params']['epochs'],
                                  callbacks = callbacks,
                                  validation_data = valid_set,
                                  validation_steps = len(valid_set)
                                  )