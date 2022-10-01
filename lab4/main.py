# CM2003 lab 4
import os

from unet import get_unet
from metrics import dice_coef
from analysis import learning_curves
from loss import dice_loss
from data_loading import load_data
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from skimage.io import imread
from tensorflow.keras import backend as K

if __name__ == '__main__':
    # set model parameters
    img_width = 256
    img_height = 256
    img_ch = 1

    n_classes = 1
    n_base = 8
    batch_size = 8

    epochs = 150
    learning_rate = 0.0001

    val_split = 0.2

    # set model parameters
    dropout_rate = 0
    use_batch_norm = True

    input_size = (img_width, img_height, img_ch)

    # set paths to data
    #base_path = "/DL_course_data/Lab3/X_ray"
    base_path = "X_ray"
    masks = "Mask"
    img = "Image"

    train_data_loader, val_data_loader, num_train_samples, num_val_samples = load_data(base_path=base_path,
                            img_path=img,
                            target_path=masks,
                            val_split=val_split,
                            batch_size=batch_size,
                           img_size=(img_width, img_height))
    # define model
    unet = get_unet(input_shape=input_size, n_classes=n_classes, n_base=n_base, batch_size=batch_size)
    unet.summary()
    unet.compile(optimizer=Adam(learning_rate=1e-3),
                 loss=dice_loss,
                 metrics=[dice_coef])
    unet_1b_hist = unet.fit(train_data_loader,
                        epochs=epochs,
                        steps_per_epoch=math.floor(num_train_samples/batch_size),
                        validation_data=val_data_loader,
                        validation_steps=math.floor(num_val_samples/batch_size)
                        )

    print(unet_1b_hist.history.keys())
    learning_curves(unet_1b_hist, "loss", "val_loss", "dice_coef", "val_dice_coef")

    print('finished')
    K.clear_session()
    # todo: binarize segmentation map

    # import data




