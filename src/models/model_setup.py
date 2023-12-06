from src.models.model_components.architectures import vgg16_unet
from src.models.model_components import loss_functions as loss, metrics


def get_model(get:str,n_classes:int):
    return {
        "VGG16-Unet": vgg16_unet.VGG16_Unet(n_classes=n_classes)
    }[get]
    
def get_metric(get:str):
    return {
        "IoU": metrics.IoU,
        "sensitivity":metrics.sensitivity,
        "specificity":metrics.specificity,
        "classification_metrics": metrics.classification_metrics
    }[get]
    
def get_loss(get:str):
    return {
        "DiceLoss": loss.DiceLoss()
    }[get]
    
    