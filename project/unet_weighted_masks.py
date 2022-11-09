# CM2003 project
import os

from unet import get_unet, get_unet_weighted
from metrics import dice_coef, precision, recall, jaccard
from analysis import learning_curves, segment_from_directory, segment_from_directory_weighted
from loss import dice_loss, combined_loss, weighted_loss, combined_weighted_loss
from dataloading import load_data_with_weight_mask
from tensorflow.keras.optimizers import Adam
import math
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.keras import backend as K

disable_eager_execution()

if __name__ == '__main__':
    # testing baseline model for retinal vessel segmentation
    # set paths to data
    # base_path = "/home/student/tf-lasse/project/dataset/train"
    base_path = "../project/dataset/train"
    # base_path = os.path.join("dataset", "train")
    masks = "training_masks"
    img = "training_images"
    # weight_masks = "training_masks_dilated"
    # weight_masks = "training_masks_dilated_eroded"
    # weight_masks = "training_masks_small_boundary"
    #weight_masks = "training_masks_preprocessing_3"
    weight_masks = "training_masks_small_boundary"
    weight_strength = 0.2
    # mask_vessels(base_path, 'training_masks', 'training_mask_dilated')

    # set model parameters
    img_width = 768
    img_height = 768
    img_ch = 1

    # set number of foreground classes
    n_classes = 2
    if n_classes > 1:
        binary_mask = False
    else:
        binary_mask = True

    # set batch size
    batch_size = 4

    # set validation set split ratio
    val_split = 0.2  # train using all data

    # set model parameters
    dropout_rate = 0.2
    use_batch_norm = True

    input_size = (img_width, img_height, img_ch)

    n_base = 64
    kernel = (3, 3)
    learning_rate = 0.0005
    alpha = 0.6

    # set number of epochs
    epochs = 2000

    augumentation_dict = dict(rotation_range = 10,
                              width_shift_range = 0.2,
                              height_shift_range = 0.2,
                              zoom_range = [0.1, 1.4],
                              horizontal_flip = True,
                              fill_mode = 'reflect',
                              )

    train_data_loader, val_data_loader, num_train_samples, num_val_samples = load_data_with_weight_mask(base_path=base_path,
                                                                                                        img_path=img,
                                                                                                        target_path=masks,
                                                                                                        weights_path=weight_masks,
                                                                                                        val_split=val_split,
                                                                                                        batch_size=batch_size,
                                                                                                        img_size=(img_width, img_height),
                                                                                                        augmentation_dic=augumentation_dict,
                                                                                                        binary_mask=binary_mask,
                                                                                                        num_classes=n_classes,
                                                                                                        patch_size=(128, 128))

    #print(next(train_data_loader[0])[0][1])
    #print("image gen shape:", next(train_data_loader[0])[0][0].shape)
    #print("weight map gen shape:", next(train_data_loader[0])[0][1].shape)
    #print("mask gen shape:", next(train_data_loader[0])[1].shape)

    # define model
    unet, loss_weights = get_unet_weighted(input_shape=(None, None, 1),
                                           n_classes=n_classes,
                                           n_base=n_base,
                                           dropout_rate=0.2,
                                           kernel_size=kernel,
                                           weights_path=weight_masks)
    unet.summary()
    unet.compile(optimizer=Adam(learning_rate=learning_rate),
                 loss=combined_weighted_loss(loss_weights, weight_strength, alpha),
                 metrics=[dice_coef, precision, recall, jaccard])

    unet_hist = unet.fit(train_data_loader[0],
                         epochs=epochs,
                         steps_per_epoch=math.floor(num_train_samples / batch_size),
                         validation_data=val_data_loader[0],
                         validation_steps=math.floor(num_val_samples / batch_size),
                         use_multiprocessing=False,
                         workers=1,
                         )

    unet.save('models/unet_baseline')
    # print model history keys
    print(unet_hist.history.keys())
    segment_from_directory_weighted(pred_dir="predictions", model=unet, base_dir="dataset", dir="test")
    segment_from_directory_weighted(pred_dir="predictions", model=unet, base_dir="dataset/train", dir="training_images")

    learning_curves(unet_hist, "loss", "val_loss", ["dice_coef", "precision", "recall"],
                    ["val_dice_coef", "val_precision", "val_recall"])

    K.clear_session()