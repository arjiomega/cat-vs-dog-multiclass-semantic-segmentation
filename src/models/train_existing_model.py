import datetime
import argparse
from pathlib import Path

import mlflow
import tensorflow as tf

import model_setup
from config import config
from src.data.load import load
from src.models.model_components.callbacks import plot_predict
from src.data.data_utils import load_label_dict, load_list, load_args

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_args',dest='train_args',type=str,help="json filename in config dir containing the training setup. Ex. 'setup1.json'")
    parser.add_argument('--model_uri',dest='model_uri',type=str,help="model_uri can be found in mlflow experiment run artifacts")
    args = parser.parse_args()
    
    if args.train_args and args.model_uri:
        args = load_args.load_args(config.CONFIG_DIR,args.train_args) 
        model_uri = args.model_uri
    else:
        parser.error("training setup file and model uri are both required.")

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
    
    model = mlflow.tensorflow.load_model(model_uri)
    
    img,mask = valid_set.dataset[50]
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%m%d%Y-%H%M%S")
    
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(),
                plot_predict.plot_predict_per_epoch(model,img,mask),
                tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1)
                ]
    
    model_history = model.fit(train_set,
                                  steps_per_epoch = len(train_set),
                                  epochs=args['params']['epochs'],
                                  callbacks = callbacks,
                                  validation_data = valid_set,
                                  validation_steps = len(valid_set)
                                  )