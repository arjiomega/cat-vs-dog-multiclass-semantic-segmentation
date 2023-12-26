from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from config import config


class plot_predict_per_epoch(tf.keras.callbacks.Callback):
    def __init__(self, model, sample_img, sample_mask, threshold=0.5):
        super(plot_predict_per_epoch, self).__init__()
        self.model = model
        self.threshold = threshold
        self.sample_img = self.remove_normalization(sample_img)
        self.sample_mask = sample_mask
        self.predict_input = np.expand_dims(sample_img, axis=0)

    def remove_normalization(self, img_input):
        # remove -1,1 img normalization
        return ((img_input + 1.0) * 127.5) / 255.0

    def postprocess_predict(self):
        predict_mask = self.model.predict(self.predict_input)
        predict_mask = np.where(predict_mask > self.threshold, 1, 0)
        return predict_mask

    def gen_plot(self):
        predict_mask = self.postprocess_predict()

        labels = ["img", "bg", "cat", "dog"]

        true_plot_list = [
            self.sample_img,  # Image
            self.sample_mask[..., 0].squeeze(),  # Background
            self.sample_mask[..., 1].squeeze(),  # Cat
            self.sample_mask[..., 2].squeeze(),  # Dog
        ]

        predict_plot_list = [
            self.sample_img,  # Image
            predict_mask[..., 0].squeeze(),  # Background
            predict_mask[..., 1].squeeze(),  # Cat
            predict_mask[..., 2].squeeze(),  # Dog
        ]

        fig, arr = plt.subplots(2, 4)

        for i, arr_i in enumerate(arr):
            for j, arr_j in enumerate(arr_i):
                title = "true" if i == 0 else "predict"
                plot_ = true_plot_list[j] if i == 0 else predict_plot_list[j]

                arr_j.set_title(f"{title}_{labels[j]}")
                arr_j.imshow(plot_)
                arr_j.axis("off")

        return fig

    def save_fig(self, fig, epoch):
        file_name = f"predict_plot_epoch_{epoch}.png"
        fig.savefig(fname=Path(config.FIGURE_DIR, file_name), format="png")

    def on_epoch_end(self, epoch, logs=None):
        fig = self.gen_plot()
        self.save_fig(fig, epoch)
