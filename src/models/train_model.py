import os
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf

from config import config
from src.data.data_utils import load_label_dict, load_list, load_args

from src.data.load import load

from src.experiment_tracking.track_experiment import TrackExperiment

from src.models.model_components.callbacks import plot_predict
from src.models.model_components import loss_functions as loss, metrics

import model_setup

def load_data(args):
    
    data_list = load_list.load(config.PROCESSED_IMG_DIR,config.PROCESSED_MASK_DIR)
    mask_label = load_label_dict.load(Path(config.RAW_DATA_DIR,'annotations'),'list.txt') #transfer path of list.txt

    train_set, valid_set, test_set = load.load_batches(data_list,
                                                       mask_label,
                                                       config.PROCESSED_IMG_DIR,
                                                       config.PROCESSED_MASK_DIR,
                                                       args)
    
    return train_set, valid_set, test_set

def load_model(args):
    learning_rate = args['params']['learning_rate']
    
    model = model_setup.get_model(get=args['setup']['model'],
                                  n_classes=args['setup']['n_classes'])
    
    loss = model_setup.get_loss(get=args['setup']['loss'])
    
    metrics = [model_setup.get_metric(get=metric) for metric in args['setup']['metrics']]
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=loss,
        metrics=metrics)

    return model

if __name__ == '__main__':
    
    # test first if the args contain the required arguments
    args = load_args.load_args(config.CONFIG_DIR,"args.json") 

    train_set, valid_set, test_set = load_data(args)
    model = load_model(args)

    # should be debugging dataset (dark images, light images, combined image of cat)
    img,mask = valid_set.dataset[50]
    
    experiment = TrackExperiment(tracking_uri=os.environ.get('MLFLOW_TRACKING_URI'),
                                 experiment_name=args['experiment']['experiment_name'],
                                 run_name=args['experiment']['run_name'],
                                 args=args)

    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(),
                 plot_predict.plot_predict_per_epoch(model,img,mask)
                 ]

    experiment.fit(model,train_set,valid_set,callbacks)
 
    custom_objects = {metric:model_setup.get_metric(get=metric) for metric in args['setup']['metrics']}
    custom_objects[args['experiment']['loss']] = model_setup.get_loss(get=args['setup']['loss'])
    custom_objects["classification_metrics"] = model_setup.get_metric(get="classification_metrics") # required for other metrics
 
    experiment.log_experiment(custom_objects=custom_objects, 
                              artifact_path='model', 
                              fig_path=config.FIGURE_DIR)