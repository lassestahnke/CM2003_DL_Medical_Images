# CM2003 project
import os

from loss import combined_loss
from metrics import dice_coef, precision, recall, jaccard
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from dataloading import load_data

if __name__ == '__main__':
    """
        This script can be used for loading models from the model folder and predicting the segmentation maps based on 
        files in the dataset directory. 
    """

    base_path = "../"
    model_dir = "models"
    model_name = "unet_baseline_600_epochs"

    masks = "training_masks"
    img = "training_images"

    model = load_model(os.path.join(base_path, model_dir, model_name))#, custom_objects=custom_objects)

    alpha = 0.4
    learning_rate = 1e-4
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                 loss=combined_loss(alpha=alpha),
                 metrics=[dice_coef, precision, recall, jaccard])

    base_path = "../dataset/train"
    masks = "training_masks"
    img = "training_images"

    train_data_loader, val_data_loader, num_train_samples, num_val_samples = load_data(base_path=base_path,
                                                                                       img_path=img,
                                                                                       target_path=masks,
                                                                                       val_split=0.0,
                                                                                       batch_size=1,
                                                                                       img_size=(768, 768),
                                                                                       augmentation_dic=None,
                                                                                       binary_mask=False,
                                                                                       num_classes=2,
                                                                                       patch_size=(256, 256))

    img, msk = next(train_data_loader[0])
    pred = model.predict(img)



