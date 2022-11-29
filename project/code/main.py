# CM2003 project
import os

import matplotlib.pyplot as plt

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

    base_path_model = "../"
    model_dir = "models"

    masks = "training_masks"
    img = "training_images"

    base_path = "../dataset/train"
    masks = "training_masks"
    img = "training_images"

    # setup dataloader for sample segmentations
    train_data_loader, val_data_loader, num_train_samples, num_val_samples = load_data(base_path=base_path,
                                                                                       img_path=img,
                                                                                       target_path=masks,
                                                                                       val_split=0.2,
                                                                                       batch_size=1,
                                                                                       img_size=(768, 768),
                                                                                       augmentation_dic=None,
                                                                                       binary_mask=False,
                                                                                       num_classes=2,
                                                                                       patch_size=(256, 256))

    # setup binary dataloader for sample segmentations
    train_data_loader_bin, val_data_loader_bin, num_train_samples_bin, num_val_samples_bin = load_data(
        base_path=base_path,
        img_path=img,
        target_path=masks,
        val_split=0.2,
        batch_size=1,
        img_size=(768, 768),
        augmentation_dic=None,
        binary_mask=False,
        num_classes=2,
        patch_size=(256, 256))

    alpha = 0.4
    learning_rate = 1e-4

    # load baseline model
    baseline_model = load_model(os.path.join(base_path_model, model_dir, "unet_baseline"), compile=False)
    baseline_model.compile(optimizer=Adam(learning_rate=learning_rate),
                 loss=combined_loss(alpha=alpha),
                 metrics=[dice_coef, precision, recall, jaccard])

    # sample prediction of baseline model
    img, msk = next(val_data_loader[0])
    pred = baseline_model.predict(img)
    plt.imshow(pred[0,:,:,1]) # plot veins
    plt.show()
    plt.imshow(pred[0, :, :, 2])  # plot vessels
    plt.show()

    # load binary model
    binary_model = load_model(os.path.join(base_path_model, model_dir, "unet_baseline_binary"), compile=False)
    binary_model.compile(optimizer=Adam(learning_rate=learning_rate),
                 loss=combined_loss(alpha=alpha),
                 metrics=[dice_coef, precision, recall, jaccard])

    # sample prediction binary model
    img_bin, msk_bin = next(val_data_loader[0])
    pred_bin = binary_model.predict(img_bin)
    plt.imshow(pred_bin[0,:,:,1]) # plot Vessels
    plt.show()


