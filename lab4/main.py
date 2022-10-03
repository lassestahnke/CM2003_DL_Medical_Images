# CM2003 lab 4
import os

from unet import get_unet
from metrics import dice_coef, precision, recall
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
    img_ch = 3

    n_classes = 1
    n_base = 8
    batch_size = 8

    epochs = 80
    learning_rate = 0.0001

    val_split = 0.2

    # set model parameters
    dropout_rate = 0.2
    use_batch_norm = True

    augumentation_dict = dict(rotation_range = 10,
                              width_shift_range = 0.1,
                              height_shift_range = 0.1,
                              zoom_range = 0.2,
                              horizontal_flip = True,
                              fill_mode = 'constant',
                              cval = 0)

    input_size = (img_width, img_height, img_ch)

    # set paths to data
    #base_path = "/DL_course_data/Lab3/X_ray"
    #base_path = "X_ray"
    base_path = "CT"
    masks = "Mask"
    img = "Image"

    train_data_loader, val_data_loader, num_train_samples, num_val_samples = load_data(base_path=base_path,
                            img_path=img,
                            target_path=masks,
                            val_split=val_split,
                            batch_size=batch_size,
                            img_size=(img_width, img_height),
                            augmentation_dic = None,
                            binary_mask=True)
    # define model
    unet = get_unet(input_shape=input_size, n_classes=n_classes, n_base=n_base, dropout_rate=0.2)
    unet.summary()
    unet.compile(optimizer=Adam(learning_rate=learning_rate),
                 loss=dice_loss,
                 #loss=tf.keras.losses.BinaryCrossentropy(),
                 metrics=[dice_coef, precision, recall])
    unet_hist = unet.fit(train_data_loader,
                        epochs=epochs,
                        steps_per_epoch=math.floor(num_train_samples/batch_size),
                        validation_data=val_data_loader,
                        validation_steps=math.floor(num_val_samples/batch_size)
                        )

    print(unet_hist.history.keys())
    learning_curves(unet_hist, "loss", "val_loss",
                    ["dice_coef", "precision", "recall"],
                    ["val_dice_coef", "val_precision", "val_recall"])
    unet.save('models/unet_3_dice')

    print('finished')
    K.clear_session()

    # import data




