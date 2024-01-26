import datetime


def saveModelCheckPoint():
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = f"test_best_model_trainable224-{timestamp}.h5"
    return str(file_name)
