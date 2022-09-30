# CM2003 lab 4
from unet import *
from metrics import *
from data_loading import load_data
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
import tensorflow as tf
import math

if __name__ == '__main__':
    # set model parameters
    img_width = 128
    img_height = 128
    img_ch = 1
    input_size = (img_width, img_height, img_ch)
    batch_size = 8
    learning_rate = 0.0001
    n_classes = 1
    n_base = 64

    # set paths to data
    #base_path = "/DL_course_data/Lab3/X_ray"
    base_path = "X_ray"
    masks = "Mask"
    img = "Image"

    data_loader, len_data = load_data(base_path=base_path,
                            img_path=img,
                            target_path=masks,
                            batch_size=batch_size,
                           img_size=(img_width, img_height))
    # define model
    unet = get_unet(input_shape=input_size, n_classes=n_classes, n_base=n_base, batch_size=batch_size)
    unet.summary()
    unet.compile(optimizer=Adam(learning_rate=1e-3),
                 loss=tf.keras.losses.BinaryCrossentropy(),
                 metrics=[dice_coef])
    unet.fit(data_loader,
             epochs=10,
             steps_per_epoch=batch_size)


    print('finished')


    # import data




