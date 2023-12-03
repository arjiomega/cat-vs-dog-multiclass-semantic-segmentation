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
from src.models.model_components.architectures import vgg16_unet

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
    learning_rate = args['learning_rate']
    
    model = vgg16_unet.VGG16_Unet(n_classes = 3)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=loss.DiceLoss(),
        metrics=[metrics.IoU,metrics.sensitivity,metrics.specificity])

    return model

if __name__ == '__main__':
    
    args = load_args.load_args(config.CONFIG_DIR,"args.json") 

    train_set, valid_set, test_set = load_data(args)
    model = load_model(args['params'])

    # should be debugging dataset (dark images, light images, combined image of cat)
    img,mask = valid_set.dataset[50]

    # start_run before training can be a bad practice according to mlflow (look documentation)
    # don't log predict plot real time or look for alternatives
    experiment = TrackExperiment(tracking_uri=os.environ.get('MLFLOW_TRACKING_URI'),
                                 experiment_name="Cat vs Dog Semantic Segmentation",
                                 run_name="Test run mlflow setup",
                                 args=args)

    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(),
                 plot_predict.plot_predict_per_epoch(model,img,mask)
                 ]

    experiment.fit(model,train_set,valid_set,callbacks)
 
    custom_objects = {
        "DiceLoss":loss.DiceLoss(),
        "IoU":metrics.IoU,
        "classification_metrics":metrics.classification_metrics,
        "sensitivity":metrics.sensitivity,
        "specificity":metrics.specificity
    }
 
    experiment.log_experiment(custom_objects=custom_objects, 
                              artifact_path='model', 
                              fig_path=config.FIGURE_DIR)
 