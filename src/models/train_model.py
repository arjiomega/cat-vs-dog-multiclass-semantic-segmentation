import os
import datetime
import argparse
from pathlib import Path

import mlflow
import tensorflow as tf

import model_setup
from config import config
from src.data.load import load
from src.models.model_components.callbacks import plot_predict
from src.experiment_tracking.track_experiment import TrackExperiment
from src.data.data_utils import load_label_dict, load_list, load_args

def load_data(args):
    
    data_list = load_list.load(config.PROCESSED_IMG_DIR,config.PROCESSED_MASK_DIR)
    mask_label = load_label_dict.load(Path(config.RAW_DATA_DIR,'annotations'),'list.txt') #transfer path of list.txt

    train_set, valid_set, test_set = load.load_batches(data_list,
                                                       mask_label,
                                                       config.PROCESSED_IMG_DIR,
                                                       config.PROCESSED_MASK_DIR,
                                                       args)
    
    return train_set, valid_set, test_set

def load_model(args,model=None):

    if not model:
        model = model_setup.get_model(get=args['setup']['model'],
                                    n_classes=args['setup']['n_classes'])
    
    loss = model_setup.get_loss(get=args['setup']['loss'])
    
    metrics = [model_setup.get_metric(get=metric) for metric in args['setup']['metrics']]
    
    model.compile(optimizer=tf.keras.optimizers.Adam(args['params']['learning_rate']),
        loss=loss,
        metrics=metrics)

    return model

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_args',dest='train_args',type=str,help="json filename in config dir containing the training setup. Ex. 'setup1.json'")
    parser.add_argument('--model_uri',dest='model_uri',type=str,help="model_uri can be found in mlflow experiment run artifacts")
    parser.add_argument('--tune_layers',dest='tune_layers',type=int,help="number of layers to be unfreezed starting from the end of pretrained model")
    input_args = parser.parse_args()
    
    # if train model only
    if input_args.train_args:
        args = load_args.load_args(config.CONFIG_DIR,input_args.train_args)
    else:
        parser.error("training setup file is required.")

    # if train existing model
    if input_args.model_uri:
        TRAIN_EXISTING_MODEL = True
        model_uri = input_args.model_uri
        
        # Load previously trained model from mlflow experiment run artifact
        model = mlflow.tensorflow.load_model(model_uri)
            
    # if fine tuning a model
    if input_args.tune_layers:
        LAYERS_TO_TUNE = input_args.tune_layers
        pretrained_model = model.layers[1]
        
        # loop starting from the end
        for layer in pretrained_model.layers[::-1][:LAYERS_TO_TUNE]:
            layer.trainable = True
            
        model = load_model(args,model)
        
    train_set, valid_set, test_set = load_data(args)
    model = load_model(args)

    # should be debugging dataset (dark images, light images, combined image of cat)
    img,mask = valid_set.dataset[50]
    
    experiment = TrackExperiment(tracking_uri=os.environ.get('MLFLOW_TRACKING_URI'),
                                experiment_name=args['experiment']['experiment_name'],
                                run_name=args['experiment']['run_name'],
                                args=args)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%m%d%Y-%H%M%S")
    
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(),
                plot_predict.plot_predict_per_epoch(model,img,mask),
                tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1)
                ]
    
    experiment.fit(model,train_set,valid_set,callbacks)
    experiment.evaluate(test_set)

    custom_objects = {metric:model_setup.get_metric(get=metric) for metric in args['setup']['metrics']}
    custom_objects[args['setup']['loss']] = model_setup.get_loss(get=args['setup']['loss'])
    custom_objects["classification_metrics"] = model_setup.get_metric(get="classification_metrics") # required for other metrics

    experiment.log_experiment(custom_objects=custom_objects, 
                            artifact_path='model', 
                            fig_path=config.FIGURE_DIR)