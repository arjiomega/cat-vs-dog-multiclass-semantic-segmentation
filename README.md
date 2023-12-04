Cat and Dog Semantic Segmentation
==============================

Prerequisites
------------
- Python
- Tensorflow
- in requirements.txt
- add PYTHONPATH env variable for easy access of python files from different directory. [(source)](https://www.geeksforgeeks.org/pythonpath-environment-variable-in-python/)


Experiment Tracking
------------
- Use the mlflow tracking provided by dagshub

Data Version Control
------------
- mention setup and that s3 works without a problem

Cat vs Dog Dataset
------------
[OXFORD-IIIT PET Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

![pet dataset image](https://www.robots.ox.ac.uk/~vgg/data/pets/pet_annotations.jpg)



Project Organization
------------

    ├── LICENSE
    ├── Makefile          
    ├── README.md    
    ├── data (DONE)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── raw.dvc        <- DVC file that tracks the raw data
    │   └── raw            <- The original, immutable data dump
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │   └── metrics.txt    <- Relevant metrics after evaluating the model.
    │   └── training_metrics.txt    <- Relevant metrics from training the model.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data            <- Scripts to download or generate data
    │   │   ├── data_utils  <- load file names for training and postprocessing
    │   │   ├── load        <- load train, valid, test datasets for training
    │   │   └── make_dataset.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── model_components <- models, callbacks, custom loss functions, custom metrics
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    ├── .pre-commit-config.yaml  <- pre-commit hooks file with selected hooks for the projects.
    ├── dvc.lock           <- The version definition of each dependency, stage, and output from the 
    │                         data pipeline.
    └── dvc.yaml           <- Defining the data pipeline stages, dependencies, and outputs.


--------
