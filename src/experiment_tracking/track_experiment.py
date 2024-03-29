import os
from pathlib import Path

import mlflow
import numpy as np

from config import config
from .debug_performance import DebuggingReportGenerator


class TrackExperiment:
    def __init__(self, tracking_uri, experiment_name, run_name, args):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name

        self.metric_labels = args["setup"]["metrics"]
        self.epochs = args["params"]["epochs"]
        self.params = args["params"]
        self.tags = args["tags"]

    def fit(self, model, train_set, valid_set, callbacks):
        model_history = model.fit(
            train_set,
            steps_per_epoch=len(train_set),
            epochs=self.epochs,
            callbacks=callbacks,
            validation_data=valid_set,
            validation_steps=len(valid_set),
        )
        self.model = model
        self.model_signature_input, _ = valid_set.dataset[50]
        self.metrics = model_history.history

    def evaluate(self, test_set):
        # results = [loss, metrics1, metrics2, metrics_n]
        results = self.model.evaluate(test_set)

        self.test_metrics = {
            f"test_{label}": score
            for label, score in zip(self.metric_labels, results[1:])
        }
        self.test_metrics["test_loss"] = results[0]

    def log_experiment(self, custom_objects, artifact_path, fig_path):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name=self.run_name):
            self.log_metrics(self.metrics, self.epochs)
            self.log_test_metrics()
            self.log_params(self.params)
            self.log_tags(self.tags)
            self.log_debug_performance()
            self.log_model(self.model, custom_objects, artifact_path)

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metrics(self, metrics, epochs):
        for i in range(epochs):
            for metric, score in metrics.items():
                mlflow.log_metric(metric, score[i], step=i)

    def log_test_metrics(self):
        mlflow.log_metrics(self.test_metrics)

    def log_tags(self, tags):
        mlflow.set_tags(tags)

    def log_debug_performance(self):
        debugging = DebuggingReportGenerator(
            debugging_set_path=config.DEBUGGING_SET_PATH, model=self.model
        )
        debugging.gen_report()
        debugging.gen_summary_fig()
        debugging.report_to_json(savepath=config.FIGURE_DIR)

        mlflow.log_artifacts(config.FIGURE_DIR, "debugging_set_report")

        debugging.clean_gen_figures()

    def load_model_signature(self, img, model):
        input_signature = np.expand_dims(img, axis=0)
        signature = mlflow.models.infer_signature(
            input_signature, model.predict(input_signature)
        )

        return signature

    def log_model(self, model, custom_objects, artifact_path):
        signature = self.load_model_signature(self.model_signature_input, self.model)
        mlflow.tensorflow.log_model(
            model,
            custom_objects=custom_objects,
            artifact_path=artifact_path,
            signature=signature,
            conda_env={
                "name": "mlflow-env",
                "channels": ["conda-forge"],
                "dependencies": [
                    "python=3.10",
                    "pip=23.3.1",
                    {"pip": [
                        "mlflow==2.9.2",
                        "tensorflow==2.15.0.post1"],}
                ],
            }
        )
