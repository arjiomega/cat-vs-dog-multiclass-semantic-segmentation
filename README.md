# Cat and Dog Semantic Segmentation

 Welcome to my Cat and Dog Semantic Segmentation project!

## Project Introduction

### Overview:
This project focuses on semantic segmentation to accurately identify and differentiate cats and dogs within images. While there are several existing semantic segmentation datasets like Cityscapes, this project stands out by offering a unique opportunity to easily implement debugging set, a concept highlighted by [Cassie Kozyrkov](https://www.youtube.com/watch?v=BynidpRMZkc&t=1s) from Google, easily as the different breeds of cats and dogs were already provided.

### Why Cats and Dogs?
Admittedly, pursuing a semantic segmentation project centered around cats and dogs might seem unconventional. However, this choice presents a valuable opportunity to showcase the implementation of debugging sets in machine learning models. Additionally, it allows us to address specific challenges easily such as varied lighting conditions and diverse breeds present in the dataset, which are crucial factors influencing model performance.

### Objectives:
- Implement debugging set
- ML Experiment Tracking
- Data Versioning
- Model deployment
- Annotating new data using label studio
- Continuous Integration Continuous Deployment


## Environment variables
Make sure you are inside the project directory
```shell
export MLFLOW_TRACKING_URI=<mlflow_tracking_uri> \
export MLFLOW_TRACKING_USERNAME=<username> \
export MLFLOW_TRACKING_PASSWORD=<password> \
export PYTHONPATH=$(pwd)
```

## Usage

### 1. Generate preprocessed data for training from raw and new data
```shell
python src/data/make_dataset.py
```
</br>

Generated Preprocessed Data Folder Structure
```
data/
├──── raw/
└──── preprocessed/ 
        ├── train_set
        │   ├── images
        │   └── masks
        │
        ├── valid_set
        │   ├── images
        │   └── masks
        │
        ├── test_set
        │   ├── images
        │   └── masks
        │
        └── debugging_set
```
</br>

Debugging set was split this way for easy generation of report regarding the model performance
```
debugging_set/
├──── cats/
│       ├── images
│       │       ├── cat_breed_1.jpg
│       │       ├── cat_breed_2.jpg
│       │       ├── cat_breed_3.jpg
│       │       └── ...
│       │
│       └── masks
│               ├── cat_breed_1.png
│               ├── cat_breed_2.png
│               ├── cat_breed_3.png
│               └── ...
│
├──── dogs/
│       ├── images
│       │       ├── dog_breed_1.jpg
│       │       ├── dog_breed_2.jpg
│       │       ├── dog_breed_3.jpg
│       │       └── ...
│       │
│       └── masks
│               ├── dog_breed_1.png
│               ├── dog_breed_2.png
│               ├── dog_breed_3.png
│               └── ...
│
└──── cats_and_dogs/
        ├── images
        │       ├── cat_and_dog_1.jpg
        │       ├── cat_and_dog_2.jpg
        │       ├── cat_and_dog_3.jpg
        |       └── ...
        │
        └── masks
                ├── cat_and_dog_1.png
                ├── cat_and_dog_2.png
                ├── cat_and_dog_3.png
                └── ... 
```


### 2. Training

1. train model using setup located in config directory
```shell
python src/models/train_model.py --train_args setup.json
```

2. train existing model in mlflow artifacts
```shell
python src/models/train_model.py --train_args setup.json --model_uri runs:/<run_id>/model
```

3. fine tune existing model by setting pretrained model layers to trainable
```shell
python src/models/train_model.py --train_args setup.json --model_uri runs:/<run_id>/model'
```

### 3. Prediction
plot of the prediction will be shown temporarily as save_path is not yet implemented
```shell
python src/models/predict_model.py --model_uri runs:/<run_id>/model --img_url image_url
```
<br/>

example run predicting cat and dog in the same image
```shell
python src/models/predict_model.py --img_url https://khpet.com/cdn/shop/articles/introducing-a-dog-to-a-cat-home_800x800.jpg?v=1593020063 --model_uri runs:/<run_id>/model
```
![prediction output](https://imgur.com/a/OzS50w5)

The prediction shown above is a great example on the importance of using debugging set so we can see the model's performance on difference scenario since the model is only trained on single cat or dog per image. 

Debugging set can be obtained from the training set or new custom examples which we would like to monitor such as:
- single cats
- single dogs
- combination of cat(s) and dog(s)
- cats and/or dogs in different kinds of environment such as dark area, noisy images like when they are in forest or several other stuff going on

For further information regarding debugging set, Cassie Kozyrkov has a great video about it on [youtube](https://www.youtube.com/watch?v=BynidpRMZkc).

## Experiment Tracking
- Use the mlflow tracking provided by dagshub

## Data Version Control
- Use DVC S3 to prevent issues when doing `dvc push` (may be dagshub specific problem)

## Cat vs Dog Dataset
[OXFORD-IIIT PET Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

The base dataset used is the PET Dataset from Oxford. Each of the mask does not differentiate cats or dogs only the border and if a cat or dog exists within that mask.

![pet dataset image](https://www.robots.ox.ac.uk/~vgg/data/pets/pet_annotations.jpg)

------------