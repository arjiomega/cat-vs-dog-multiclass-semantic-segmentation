import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import tensorflow as tf

from src.models.model_components import metrics

tf.random.set_seed(5)
np.random.seed(5)


def test_IoU_output():
    """
    Test the Intersection over Union (IoU) metric calculation.

    y_true =    sample 1 |      1 0 0       0 0 0       1 0 1       1 1 0
                                1 1 1       0 0 1       0 1 0       0 1 1
                                class 1     class 2     class 3     class 4

                sample 2 |      0 1 1       1 0 0       1 0 0       1 0 1
                                1 0 0       0 1 0       1 0 1       1 1 0
                                class 1     class 2     class 3     class 4

    y_pred =    sample 1 |      0 0 1       0 1 0       0 1 0       0 1 0
                                1 1 0       1 1 1       1 1 0       0 0 0
                                class 1     class 2     class 3     class 4

                sample 2 |      1 1 0       1 1 1       1 1 1       0 1 1
                                1 0 1       1 0 0       1 0 1       0 1 0
                                class 1     class 2     class 3     class 4
    """

    y_true = tf.cast(np.random.randint(0, 2, (2, 2, 3, 4)), dtype=tf.float32)
    y_pred = tf.cast(np.random.randint(0, 2, (2, 2, 3, 4)), dtype=tf.float32)

    iou = metrics.IoU(y_true, y_pred)

    assert round(float(iou.numpy()), 4) == 0.3375, "Wrong implementation if IoU"


def test_IoU_shape():
    pass
