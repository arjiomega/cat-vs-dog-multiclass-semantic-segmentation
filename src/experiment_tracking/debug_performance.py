import os
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from config import config
from src.data.load import load
from src.data.preprocess import preprocess
from src.data.load import dataloader, batchloader

from src.models.model_utils import model_utils


class DebuggingReportGenerator:
    def __init__(self, debugging_set_path, model):
        self.model = model
        self.debugging_set_path = debugging_set_path
        self.model_performance = None
        self.data_dict = self.gen_data_dict()

    def gen_data_dict(self) -> dict[str : dict[str : list[str]]]:
        conditions_list = os.listdir(self.debugging_set_path)

        data_dict = {
            condition: {
                "images": [
                    image
                    for image in os.listdir(
                        Path(self.debugging_set_path, condition, "images")
                    )
                ],
                "masks": [
                    image
                    for image in os.listdir(
                        Path(self.debugging_set_path, condition, "masks")
                    )
                ],
            }
            for condition in conditions_list
        }

        return data_dict

    def gen_score(self, condition, filename) -> dict[str:float]:
        image_path = Path(self.debugging_set_path, condition, "images")
        mask_path = Path(self.debugging_set_path, condition, "masks")

        image_filename = filename + ".jpg"
        mask_filename = filename + ".png"

        dataset = dataloader.DataLoader(
            img_dir=image_path,
            mask_dir=mask_path,
            img_list=[image_filename],
            mask_list=[mask_filename],
            n_classes=3,
        )

        batchset = batchloader.BatchLoader(dataset=dataset, batch_size=1, shuffle=False)

        eval = self.model.evaluate(batchset)

        metrics_list = ["loss", "IoU", "sensitivity", "specificity"]

        return {metric: score for metric, score in zip(metrics_list, eval)}

    def gen_plot(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        predict_mask: np.ndarray,
        metrics: dict[str:float],
        image_name: str,
    ) -> matplotlib.figure.Figure:
        TEXT_SPACING = 30

        image = preprocess.remove_normalization(image)
        predict_mask = np.argmax(predict_mask[0], axis=-1)

        image_name = image_name.split("_")
        image_name = "\n".join(image_name)

        plot_list = [image, mask, predict_mask]
        label_list = [image_name, "ground\ntruth", "predict"]

        fig, arr = plt.subplots(1, 3, figsize=(7, 4))

        for i, arr_i in enumerate(arr):
            arr_i.imshow(plot_list[i])
            arr_i.set_title(label_list[i])
            arr_i.axis("off")

            if label_list[i] == "predict":
                text_pos_y = 250
                for metric, score in metrics.items():
                    text_ = f"{metric}: {score:.2f}"
                    arr_i.text(
                        112,
                        text_pos_y,
                        text_,
                        horizontalalignment="center",
                        verticalalignment="center",
                    )
                    text_pos_y += TEXT_SPACING

        fig.subplots_adjust(wspace=0, hspace=0)

        return fig

    def gen_figure(self, condition, filename, score):
        image_path = Path(self.debugging_set_path, condition, "images")
        mask_path = Path(self.debugging_set_path, condition, "masks")
        savepath = Path(config.FIGURE_DIR, condition)
        savepath.mkdir(parents=True, exist_ok=True)

        image_filename = filename + ".jpg"
        mask_filename = filename + ".png"

        image = load.load_image(
            filepath=image_path,
            filename=image_filename,
            preprocess_list=["resize", "normalize"],
            shape=(224, 224),
            normalize_range=(-1, 1),
        )
        mask = load.load_mask(
            filepath=mask_path,
            filename=mask_filename,
            preprocess_list=["resize"],
            shape=(224, 224),
        )

        predict_mask = model_utils.predict(model=self.model, img=image)

        fig = self.gen_plot(image, mask, predict_mask, score, image_name=filename)
        fig.savefig(fname=Path(savepath, filename))

    def gen_report(self):
        model_performance = {}

        for condition in self.data_dict:
            sample_list = os.listdir(Path(self.debugging_set_path, condition, "images"))
            sample_list = [sample.split(".")[0] for sample in sample_list]

            for sample in sample_list:
                score = self.gen_score(condition=condition, filename=sample)
                self.gen_figure(condition=condition, filename=sample, score=score)

                model_performance[condition] = model_performance.get(condition, {})
                model_performance[condition][sample] = score

        self.model_performance = model_performance

    def gen_summary_fig(self):
        for condition, condition_dict in self.model_performance.items():
            samples = [sample for sample in condition_dict]
            samples_IoU = [
                metrics_dict["IoU"] for metrics_dict in condition_dict.values()
            ]

            fig, arr = plt.subplots(figsize=(16, 9))

            arr.barh(samples, samples_IoU)

            for s in ["top", "bottom", "left", "right"]:
                arr.spines[s].set_visible(False)

            arr.xaxis.set_ticks_position("none")
            arr.yaxis.set_ticks_position("none")
            arr.xaxis.set_tick_params(pad=5)
            arr.yaxis.set_tick_params(pad=10)

            arr.grid(
                visible=True, color="grey", linestyle="-.", linewidth=0.5, alpha=0.5
            )

            arr.invert_yaxis()

            arr.set_title(
                f"model performance on {condition}",
                loc="left",
            )

            fig.savefig(fname=Path(config.FIGURE_DIR, f"summary_{condition}"))

    def report_to_json(self, savepath):
        savepath = Path(savepath, "model_performance_summary.json")
        with open(savepath, "w") as file:
            json.dump(self.model_performance, file, indent=4)

    def clean_gen_figures(self):
        for filename in os.listdir(config.FIGURE_DIR):
            filepath = Path(config.FIGURE_DIR, filename)

            if os.path.isdir(filename) and filename != ".gitkeep":
                shutil.rmtree(filepath)
            elif os.path.isfile(filename) and filename != ".gitkeep":
                os.remove(filepath)
