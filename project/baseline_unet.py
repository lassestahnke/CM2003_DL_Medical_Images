# CM2003 project
import os

from unet import get_unet
from metrics import dice_coef, precision, recall, jaccard
from analysis import learning_curves, segment_from_directory
from loss import dice_loss, combined_loss
from dataloading import load_data
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from skimage.io import imread
from tensorflow.keras import backend as K
import json
import numpy as np

#gpu_devices = tf.config.experimental.list_physical_devices('GPU')
#gpu_device = tf.test.gpu_device_name()
#tf.config.experimental.set_memory_growth(gpu_device, True)
#tf.config.gpu.set_per_process_memory_growth(True)

if __name__ == '__main__':
    # testing baseline model for retinal vessel segmentation
    # set paths to data
    #base_path = "/home/student/tf-lasse/project/dataset/train"
    base_path = "../project/dataset/train"
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
    val_split = 0.2 # train using all data

    # set model parameters
    dropout_rate = 0.2
    use_batch_norm = True

    input_size = (img_width, img_height, img_ch)

    n_base = 64
    kernel = (5,5)
    learning_rate = 0.0001
    alpha = 0.6

    # set number of epochs
    epochs = 100

    train_data_loader, val_data_loader, num_train_samples, num_val_samples = load_data(base_path=base_path,
                                                                                       img_path=img,
                                                                                       target_path=masks,
                                                                                       val_split=val_split,
                                                                                       batch_size=batch_size,
                                                                                       img_size=(img_width, img_height),
                                                                                       augmentation_dic=None,
                                                                                       binary_mask=binary_mask,
                                                                                       num_classes=n_classes,
                                                                                       patch_size=(256,256))

    print(next(train_data_loader[0])[0].shape)

    # define model
    unet = get_unet(input_shape=(None, None, 1),
                    n_classes=n_classes,
                    n_base=n_base,
                    dropout_rate=0.2,
                    kernel_size=kernel)
    unet.summary()
    unet.compile(optimizer=Adam(learning_rate=learning_rate),
                 loss=combined_loss(alpha=alpha),
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
    segment_from_directory(pred_dir="predictions", model=unet, base_dir="dataset", dir="test")