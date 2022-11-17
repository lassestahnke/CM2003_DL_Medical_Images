# CM2003 project
import os

from ResUnet import get_ResUnet
from metrics import dice_coef, precision, recall, jaccard
from analysis import learning_curves, segment_from_directory
from loss import dice_loss, combined_loss, combined_loss_class_dice
from dataloading import load_data
from tensorflow.keras.optimizers import Adam
import math

if __name__ == '__main__':
    # testing baseline model for retinal vessel segmentation
    # set paths to data
    # base_path = "/home/student/tf-lasse/project/dataset/train"
    base_path = "../dataset/train"
    masks = "training_masks"
    img = "training_images"

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
    dropout_rate = 0.4
    use_batch_norm = True

    input_size = (img_width, img_height, img_ch)

    n_base = 64
    kernel = (3, 3)
    learning_rate = 0.0005
    alpha = 0.4
    beta = 0.3

    # set number of epochs
    epochs = 500

    augmentation_dict = dict(rotation_range = 10,
                              width_shift_range = 0.2,
                              height_shift_range = 0.2,
                              zoom_range = [0.1, 1.4],
                              horizontal_flip = True,
                              fill_mode = 'reflect')

    train_data_loader, val_data_loader, num_train_samples, num_val_samples = load_data(base_path=base_path,
                                                                                       img_path=img,
                                                                                       target_path=masks,
                                                                                       val_split=val_split,
                                                                                       batch_size=batch_size,
                                                                                       img_size=(img_width, img_height),
                                                                                       augmentation_dic=augmentation_dict,
                                                                                       binary_mask=binary_mask,
                                                                                       num_classes=n_classes,
                                                                                       patch_size=(256, 256))

    print(next(train_data_loader[0])[0].shape)

    # define model
    unet = get_ResUnet(input_shape=(None, None, 1),
                    n_classes=n_classes,
                    n_base=n_base,
                    dropout_rate=0,
                    kernel_size=kernel)
    unet.summary()
    unet.compile(optimizer=Adam(learning_rate=learning_rate),
                 loss=combined_loss_class_dice(alpha=alpha, beta=beta),
                 metrics=[dice_coef, precision, recall, jaccard])
    unet_hist = unet.fit(train_data_loader[0],
                         epochs=epochs,
                         steps_per_epoch=math.floor(num_train_samples / batch_size),
                         validation_data=val_data_loader[0],
                         validation_steps=math.floor(num_val_samples / batch_size),
                         use_multiprocessing=False,
                         workers=1,
                         )
    unet.save('models/ResUnet_test')
    # print model history keys
    print(unet_hist.history.keys())
    #segment_from_directory(pred_dir="predictions", model=unet, base_dir="dataset", dir="test")
    #segment_from_directory(pred_dir="predictions", model=unet, base_dir="dataset/train", dir="training_images")
    segment_from_directory(pred_dir="../code/predictions", model=unet, base_dir="../dataset", dir="test")
    segment_from_directory(pred_dir="../code/predictions", model=unet, base_dir="../dataset/train",
                           dir="training_images")
    learning_curves(unet_hist, "loss", None, ["dice_coef", "precision", "recall"],
                    None)

    learning_curves(unet_hist, "loss", "val_loss",
                   ["dice_coef", "precision", "recall", "jaccard"],
                    ["val_dice_coef", "val_precision", "val_recall", "val_jaccard"])
