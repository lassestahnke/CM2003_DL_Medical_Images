# CM2003 project
import os

from loss import combined_loss
from metrics import dice_coef, precision, recall, jaccard
from tensorflow.keras.models import load_model

if __name__ == '__main__':
    """
        This script can be used for loading models from the model folder and predicting the segmentation maps based on 
        files in the dataset directory. 
    """

    base_path = "../"
    model_dir = "models"
    model_name = "unet_baseline"

    masks = "training_masks"
    img = "training_images"

    # specify custom metrics and losses:
    custom_objects = {"combined_loss": combined_loss(0.4),# "metrics": [dice_coef, precision, recall, jaccard]}
                      "dice_coef": dice_coef,
                      "precison": precision,
                      "recall": recall,
                      "jaccard": jaccard}
    model = load_model(os.path.join(base_path, model_dir, model_name), compile=False)#, custom_objects=custom_objects)









