import os
from pathlib import Path

import mlflow
import numpy as np
from PIL import Image

class TrackExperiment:
    def __init__(self,tracking_uri,experiment_name,run_name,args):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        
        self.epochs = args['params']['epochs']
        self.params = args['params']
        self.tags = args['tags']
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
    def fit(self,model,train_set,valid_set,callbacks):
        model_history = model.fit(train_set,
                                  steps_per_epoch = len(train_set),
                                  epochs=self.epochs,
                                  callbacks = callbacks,
                                  validation_data = valid_set,
                                  validation_steps = len(valid_set)
                                  )
        
        self.model = model
        self.model_signature_input, _ = valid_set.dataset[50] # change this to debug set
        self.metrics = model_history.history
        
    def log_experiment(self,custom_objects,artifact_path,fig_path):
        with mlflow.start_run(run_name=self.run_name):
            self.log_metrics(self.metrics,self.epochs)
            self.log_params(self.params)
            self.log_tags(self.tags)
            self.log_figures(fig_path)
            self.log_model(self.model,custom_objects,artifact_path)
                
    def log_params(self,params):
        mlflow.log_params(params)
        
    def log_metrics(self,metrics,epochs):
        for i in range(epochs):
            for metric,score in metrics.items():
                mlflow.log_metric(metric,score[i],step=i)
        
    def log_tags(self,tags):
        mlflow.set_tags(tags)
        
    def log_figures(self,fig_path):
        figure_list = os.listdir(fig_path)
        
        for figure in figure_list:
            file = Path(fig_path,figure)
            mlflow.log_artifact(file,f'predict_plots/{figure}')
        
    def load_model_signature(self,img,model):
        input_signature = np.expand_dims(img, axis=0)
        signature = mlflow.models.infer_signature(input_signature,model.predict(input_signature))
        
        return signature
    
    def log_model(self,model,custom_objects,artifact_path):
        signature = self.load_model_signature(self.model_signature_input,self.model)
        mlflow.tensorflow.log_model(model,
                                    custom_objects=custom_objects,
                                    artifact_path=artifact_path,
                                    signature=signature)
        
    
        