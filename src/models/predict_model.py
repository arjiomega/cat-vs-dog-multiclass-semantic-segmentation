"""Predict using model from MLFlow artifacts"""
import urllib
import argparse

import cv2
import mlflow
import numpy as np
import matplotlib.pyplot as plt

def preprocess(img):
    #resize
    preprocess_img = cv2.resize(img, (224,224), interpolation=cv2.INTER_LINEAR)
    #normalize
    preprocess_img = (preprocess_img/127.5) - 1.0

    return preprocess_img

def remove_normalization(img):
    # remove -1,1 img normalization
    return ((img+1.0) * 127.5)/255.0

def img_loader(url):
    response = urllib.request.urlopen(url)

    img = np.asarray(bytearray(response.read()), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def predict(model,img):
    # preprocess for model prediction
    preprocess_img = np.expand_dims(img, axis=0)

    # generate mask from model
    predict_mask = model.predict(preprocess_img)

    threshold = 0.5
    predict_mask = np.where(predict_mask > threshold, 1, 0)

    return predict_mask

def gen_plot(img,predict_mask):
    sample_img = remove_normalization(img)

    labels = ['img','bg','cat','dog','all']

    test = np.argmax(predict_mask[0],axis=-1)

    predict_plot_list = [
        sample_img,                # Image
        predict_mask[...,0].squeeze(),  # Background
        predict_mask[...,1].squeeze(),  # Cat
        predict_mask[...,2].squeeze(),  # Dog
        test
    ]

    fig,arr = plt.subplots(1,5)

    for i,arr_i in enumerate(arr):
        title = "predict"
        plot_ = predict_plot_list[i]

        arr_i.set_title(f"{title}_{labels[i]}")
        arr_i.imshow(plot_)
        arr_i.axis('off')

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_uri',dest='model_uri',type=str,help="model_uri can be found in mlflow experiment run artifacts")
    parser.add_argument('--img_url',dest='img_url',type=str,help="img url that is going to be used to generate a predict mask")
    parser.add_argument('--save_path',dest='save_path',type=str,help="path where you want to save prediction mask")
    input_args = parser.parse_args()
    
    if input_args.model_uri and input_args.img_url:
        IMG_URL = input_args.img_url
        model = mlflow.tensorflow.load_model(input_args.model_uri)
    else:
        parser.error("model uri and img url are both required.")

    # load img from url
    img = img_loader(IMG_URL)
    
    # preprocess img for prediction
    preprocess_img = preprocess(img)
    
    # predict mask
    predict_mask = predict(model,preprocess_img)
    
    # generate figure
    fig = gen_plot(preprocess_img,predict_mask)
    plt.show()