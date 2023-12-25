import tensorflow as tf


def classification_metrics(y_true, y_pred, threshold=0.5):
    """
    TP -> y_true 1s and y_pred 1s
    TN -> y_true 0s and y_pred 0s
    FP -> y_true 0s and y_pred 1s
    FN -> y_true 1s and y_pred 0s
    """
    y_true = tf.cast(y_true, dtype=tf.int32)
    y_pred_adj = tf.identity(y_pred)

    y_pred_adj = tf.cast(y_pred_adj > threshold, dtype=tf.int32)

    TP = tf.size(tf.where(tf.equal(y_true + y_pred_adj, 2)))
    TN = tf.size(tf.where(tf.equal(y_true + y_pred_adj, 0)))
    FP = tf.size(tf.where(tf.equal(y_true - y_pred_adj, -1)))
    FN = tf.size(tf.where(tf.equal(y_true - y_pred_adj, 1)))

    return (TP, TN, FP, FN)


def IoU(y_true: tf.Tensor, y_pred: tf.Tensor, smooth=1e-6):
    """
    Calculates Intersection over Union (IoU) metric.

    Parameters:
    - y_true : tensor
        Ground truth labels (true masks).
    - y_pred : tensor
        Predicted labels (predicted masks).
    - smooth : float, optional
        Smoothing factor to avoid division by zero, defaults to 1e-6.

    Returns:
    - float
        IoU value for the entire batch.

    This function calculates the IoU metric to measure the similarity
    between the ground truth and predicted masks for a batch of data.
    The IoU is computed as the ratio of the intersection to the union
    of the true and predicted masks.
    """

    # calculate intersection of each classes per batch
    # shape (batchsize,n_classes)
    intersection = tf.math.reduce_sum(tf.abs(y_true * y_pred), axis=(1, 2))

    # calculate union of each classes per batch
    # shape (batchsize,n_classes)
    union = (
        tf.math.reduce_sum(y_true, axis=(1, 2))
        + tf.math.reduce_sum(y_pred, axis=(1, 2))
        - intersection
    )

    # calculate iou of each class per batch
    # shape (batchsize,n_classes)
    iou = (intersection + smooth) / (union + smooth)

    # calculate iou for each batch
    # shape (batchsize,)
    iou = tf.math.reduce_mean(iou, axis=-1)

    # calculate iou for the whole batch
    # shape ()
    iou = tf.math.reduce_mean(iou)

    return iou


def sensitivity(y_true, y_pred, threshold=0.5, smooth=1e-6):
    TP, _, _, FN = classification_metrics(y_true, y_pred, threshold)
    TP, FN = tf.cast(TP, dtype=tf.float32), tf.cast(FN, dtype=tf.float32)

    TPR = (TP + smooth) / (TP + FN + smooth)

    return TPR


def specificity(y_true, y_pred, threshold=0.5, smooth=1e-6):
    _, TN, FP, _ = classification_metrics(y_true, y_pred, threshold)
    TN, FP = tf.cast(TN, dtype=tf.float32), tf.cast(FP, dtype=tf.float32)

    TNR = (TN + smooth) / (TN + FP + smooth)

    return TNR
