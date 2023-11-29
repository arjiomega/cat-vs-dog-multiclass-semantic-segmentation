import tensorflow as tf

def classification_metrics(y_true,y_pred, threshold=0.5):
    """
    TP -> y_true 1s and y_pred 1s 
    TN -> y_true 0s and y_pred 0s
    FP -> y_true 0s and y_pred 1s
    FN -> y_true 1s and y_pred 0s
    """
    y_true = tf.cast(y_true,dtype=tf.int32)
    y_pred_adj = tf.identity(y_pred)

    y_pred_adj = tf.cast(y_pred_adj > threshold, dtype=tf.int32)

    TP = tf.size(tf.where(tf.equal(y_true + y_pred_adj, 2)))
    TN = tf.size(tf.where(tf.equal(y_true + y_pred_adj, 0)))
    FP = tf.size(tf.where(tf.equal(y_true - y_pred_adj, -1)))
    FN = tf.size(tf.where(tf.equal(y_true - y_pred_adj, 1)))

    return (TP,TN,FP,FN)

def IoU(y_true, y_pred, smooth=1e-6):
  # intersection returns a shape of (n_samples,) because everything is summed
  intersection = tf.math.reduce_sum(tf.abs(y_true * y_pred), axis=(1,2,3))
  union = tf.math.reduce_sum(y_true,axis=(1,2,3))+tf.math.reduce_sum(y_pred,axis=(1,2,3))-intersection
  iou = tf.math.reduce_mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

def sensitivity(y_true, y_pred,threshold=0.5,smooth=1e-6):
    
    TP,_,_,FN = classification_metrics(y_true,y_pred, threshold)
    TP,FN = tf.cast(TP,dtype=tf.float32),tf.cast(FN,dtype=tf.float32)

    TPR = (TP+smooth)/(TP+FN+smooth)

    return TPR

def specificity(y_true, y_pred, threshold=0.5, smooth=1e-6):
    _,TN,FP,_ = classification_metrics(y_true,y_pred, threshold)
    TN,FP = tf.cast(TN,dtype=tf.float32),tf.cast(FP,dtype=tf.float32)

    TNR = (TN+smooth)/(TN+FP+smooth)

    return TNR