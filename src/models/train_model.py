
import mlflow
import tensorflow as tf

from config import config
from src.models.model_components.callbacks import save_model_checkpoint, plot_predict
from src.models.model_components import loss_functions as loss, metrics
from src.models.model_components.architectures import vgg16_unet
from src.data.data_utils import load_data, load_specie_dict
from src.models.model_components import model_data

if __name__ == '__main__':
    """
    1. load data
    2. split by class
    3. split to train,valid,test
    4. Load_dataset
    5. Load batches
    """
    img_list, mask_list = load_data.load(config.PROCESSED_IMG_DIR,config.PROCESSED_MASK_DIR)
    mask_label = load_specie_dict.load()

    train_set, valid_set, test_set = model_data.load(img_list,mask_list,mask_label)

    with mlflow.start_run():

        epochs = 50
        learning_rate = 1e-5

        model = vgg16_unet.VGG16_Unet(n_classes = 3)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                loss=loss.DiceLoss(),
                metrics=[metrics.IoU,metrics.sensitivity,metrics.specificity])
        
        model_history = model.fit(train_set,
                                  steps_per_epoch = len(train_set),
                                  epochs = epochs,
                                  callbacks = [
                                            #tf.keras.callbacks.ModelCheckpoint( \
                                            # save_model_checkpoint.saveModelCheckPoint(), \
                                            #save_best_only=True, mode='min'), \
                                            tf.keras.callbacks.ReduceLROnPlateau(),
                                            plot_predict.plot_predict_per_epoch(model),
                                        ],
                                  validation_data = valid_set,
                                  validation_steps = len(valid_set),
                                    )