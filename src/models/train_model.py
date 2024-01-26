import os
import datetime
import argparse
from pathlib import Path

import mlflow
import tensorflow as tf

import model_setup
from config import config
from src.utils.load_args import load_args
from src.data.load import batchloader, dataloader
from src.experiment_tracking.track_experiment import TrackExperiment


class LoadModel:
    def __init__(self, model_setup, n_classes, loss_function, metrics, learning_rate):
        self.model_setup = model_setup
        self.n_classes = n_classes
        self.loss_function = loss_function
        self.metrics = metrics
        self.learning_rate = learning_rate

    def load_default(self):
        model = model_setup.get_model(get=self.model_setup, n_classes=self.n_classes)
        model = self.compile_model(model)

        return model

    def load_exist(self, model_uri):
        model = mlflow.tensorflow.load_model(model_uri)
        return model

    def load_finetune(self, model, layers_to_tune):
        pretrained_model = model.layers[1]

        # loop starting from the end
        for layer in pretrained_model.layers[::-1][:layers_to_tune]:
            layer.trainable = True

        model = self.compile_model(model)

        return model

    def compile_model(self, model):
        loss = model_setup.get_loss(get=self.loss_function)

        metrics = [model_setup.get_metric(get=metric) for metric in self.metrics]

        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss=loss,
            metrics=metrics,
        )

        return model

    def load_model(self, input_args):
        if input_args.model_uri:
            model = self.load_exist(input_args.model_uri)
        else:
            model = self.load_default()

        if input_args.tune_layers:
            model = self.load_finetune(model, input_args.tune_layers)

        return model


def load_data(args):
    dataset_list = ["train_set", "valid_set", "test_set"]
    dataset_dict = {}
    for dataset_name in dataset_list:
        images_path = Path(config.PROCESSED_DATA_DIR, dataset_name, "images")
        masks_path = Path(config.PROCESSED_DATA_DIR, dataset_name, "masks")

        dataset = dataloader.DataLoader(
            img_dir=images_path,
            mask_dir=masks_path,
            img_list=os.listdir(images_path),
            mask_list=os.listdir(masks_path),
            n_classes=3,
        )
        batch = batchloader.BatchLoader(
            dataset=dataset, batch_size=args["params"]["batch_size"], shuffle=True
        )

        dataset_dict[dataset_name] = batch

    return dataset_dict


def load_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_args",
        dest="train_args",
        type=str,
        help="json filename in config dir containing the training setup. Ex. 'setup1.json'",
    )
    parser.add_argument(
        "--model_uri",
        dest="model_uri",
        type=str,
        help="model_uri can be found in mlflow experiment run artifacts",
    )
    parser.add_argument(
        "--tune_layers",
        dest="tune_layers",
        type=int,
        help="number of layers to be unfreezed starting from the end of pretrained model",
    )
    input_args = parser.parse_args()

    if not input_args.train_args:
        parser.error("training setup file is required.")

    return input_args, parser


if __name__ == "__main__":
    # Load CLI args
    input_args, parser = load_input_args()

    # Training setup
    args = load_args(config.CONFIG_DIR, input_args.train_args)

    # Load model
    loadmodel = LoadModel(
        model_setup=args["setup"]["model"],
        n_classes=args["setup"]["n_classes"],
        loss_function=args["setup"]["loss"],
        metrics=args["setup"]["metrics"],
        learning_rate=args["params"]["learning_rate"],
    )
    model = loadmodel.load_model(input_args)

    # Load data
    dataset_dict = load_data(args)
    train_set, valid_set, test_set = (
        dataset_dict["train_set"],
        dataset_dict["valid_set"],
        dataset_dict["test_set"],
    )

    # Experiment tracking
    experiment = TrackExperiment(
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
        experiment_name=args["experiment"]["experiment_name"],
        run_name=args["experiment"]["run_name"],
        args=args,
    )

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%m%d%Y-%H%M%S")
    log_dir = Path(config.REPORTS_DIR, log_dir)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
    ]

    experiment.fit(model, train_set, valid_set, callbacks)
    experiment.evaluate(test_set)

    custom_objects = {
        metric: model_setup.get_metric(get=metric)
        for metric in args["setup"]["metrics"]
    }
    custom_objects[args["setup"]["loss"]] = model_setup.get_loss(
        get=args["setup"]["loss"]
    )
    custom_objects["classification_metrics"] = model_setup.get_metric(
        get="classification_metrics"
    )

    experiment.log_experiment(
        custom_objects=custom_objects, artifact_path="model", fig_path=config.FIGURE_DIR
    )
