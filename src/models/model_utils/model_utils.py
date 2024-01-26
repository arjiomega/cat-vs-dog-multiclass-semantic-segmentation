import numpy as np


def predict(model, img):
    THRESHOLD = 0.5

    preprocess_img = np.expand_dims(img, axis=0)

    predict_mask = model.predict(preprocess_img)
    predict_mask = np.where(predict_mask > THRESHOLD, 1, 0)

    return predict_mask
