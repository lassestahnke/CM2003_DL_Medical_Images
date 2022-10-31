# CM2003 project
import os

from unet import get_unet
from metrics import dice_coef, precision, recall, jaccard
from analysis import learning_curves
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
import open
import numpy as np

if __name__ == '__main__':
    # testing baseline model for retinal vessel segmentation
    # set paths to data
    #base_path = "/DL_course_data/Lab3/X_ray"
    #base_path = "/DL_course_data/Lab3/CT"
    #base_path = "X_ray"
    base_path = "/home/student//tf-lasse/project/dataset/train"
    masks = "training_masks"
    img = "training_images"


    # set model parameters
    img_width = 768
    img_height = 768
    img_ch = 1

    # set number of foreground classes
    n_classes = 2
    if n_classes > 1:
        binary_mask=False
    else:
        binary_mask=True

    # set batch size
    batch_size = 4

    # set number of epochs
    epochs = 1000

    # set learning rate
    learning_rate = 0.0001

    # set validation set split ratio
    val_split = 0.2

    # set model parameters
    dropout_rate = 0.2
    use_batch_norm = True

    input_size = (img_width, img_height, img_ch)

    train_data_loader, val_data_loader, num_train_samples, num_val_samples = load_data(base_path=base_path,
                                                                                       img_path=img,
                                                                                       target_path=masks,
                                                                                       val_split=val_split,
                                                                                       batch_size=batch_size,
                                                                                       img_size=(img_width, img_height),
                                                                                       augmentation_dic=None,
                                                                                       binary_mask=binary_mask,
                                                                                       num_classes=n_classes)

    # grid search settings
    n_exp = 0
    grd_srch_n_base = [16, 32, 64]
    kernel_size = [(3,3), (5,5)]
    learning_rate = [0.001, 0.0001, 0.00001]
    alphas = np.array([1, 0.8, 0.6, 0.4])

    for n_base in grd_srch_n_base:
        for kernel in kernel_size:
            for lr in learning_rate:
                for alpha in alphas:

                    # define model
                    unet = get_unet(input_shape=input_size,
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
                                        steps_per_epoch=math.floor(num_train_samples/batch_size),
                                        validation_data=val_data_loader[0],
                                        validation_steps=math.floor(num_val_samples/batch_size),
                                        )

                    # print model history keys
                    print(unet_hist.history.keys())

                    # save parameters and results to json
                    json_path = "grid_search"
                    json_file_name = "base_grid_search_{}.json".format(n_exp)

                    params = {
                        "n_base": n_base,
                        "learning_rate": learning_rate,
                        "alpha": alpha,
                        "kernel_size": kernel,
                        "augmentation": "None",
                        "epochs": epochs,
                        "loss": unet_hist.history("loss"),
                        "val_loss": unet_hist.history("val_loss"),
                        "dice_coef": unet_hist.history("dice_coef"),
                        "val_dice_coef": unet_hist.history("val_dice_coef"),
                        "precision": unet_hist.history("precision"),
                        "val_precision": unet_hist.history("val_precision"),
                        "recall": unet_hist.history("recall"),
                        "val_recall": unet_hist.history("val_recall")
                    }


                    with open(os.path.join(json_path, json_file_name), "w") as outfile:
                        json.dumps(params, outfile)
                        
                    # code run finished debug
                    print('finished experiment ', n_exp)
                    n_exp += 1


    K.clear_session()





