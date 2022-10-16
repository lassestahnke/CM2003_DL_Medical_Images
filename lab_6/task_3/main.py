# CM2003 lab 4
import os
from unet import get_unet
from metrics import dice_coef, precision, recall
from analysis import learning_curves
from loss import dice_loss, weighted_loss
from data_loading import load_data
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from skimage.io import imread
from tensorflow.keras import backend as K
from preprocessing import mask_boundaries
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

if __name__ == '__main__':
    # set model parameters
    img_width = 240
    img_height = 240
    img_ch = 1

    n_classes = 1
    if n_classes > 1:
        binary_mask=False
    else:
        binary_mask=True

    n_base = 8
    batch_size = 8
    epochs = 50
    learning_rate = 0.0001
    val_split = 0.2

    # set model parameters
    dropout_rate = 0.2
    use_batch_norm = True

    # use cross-validation:
    cross_val = 1

    augumentation_dict = dict(rotation_range = 10,
                              width_shift_range = 0.1,
                              height_shift_range = 0.1,
                              zoom_range = 0.2,
                              horizontal_flip = True,
                              fill_mode = 'constant',
                              cval = 0)

    input_size = (img_width, img_height, img_ch)
    # options for weightes loss
    weight_strength = 0.0

    # set paths to data
    base_path = "/DL_course_data/Lab3/MRI"
    masks = "Mask"
    img = "Image"
    #boundary = "Boundary"
    boundary = None




    train_data_loader, val_data_loader, num_train_samples, num_val_samples = load_data(base_path=base_path,
                            img_path=img,
                            target_path=masks,
                            boundary_path=boundary,
                            val_split=val_split,
                            batch_size=batch_size,
                            img_size=(img_width, img_height),
                            augmentation_dic = None,
                            binary_mask=binary_mask,
                            cross_val=cross_val)

    # store model(s) and histories in models list and model_histories list, respectively
    models = []
    model_histories = []
    for k in range(cross_val):
        # define model
        print("Training of ", k, "-th fold")
        # unet option for weighted loss with boundaries
        if boundary is not None:
            unet, loss_weights = get_unet(input_shape=input_size, n_classes=n_classes, n_base=n_base, dropout_rate=0.2,
                                          boundary_path=boundary)
            unet.summary()
            # compile option for weighted loss with boundaries
            unet.compile(optimizer=Adam(learning_rate=learning_rate),
                         loss=weighted_loss(loss_weights, weight_strength),
                         metrics=[dice_coef, precision, recall])
        else:
            unet = get_unet(input_shape=input_size, n_classes=n_classes, n_base=n_base, dropout_rate=0.2,
                            boundary_path=boundary)
            unet.summary()
            # compile option for normal dice loss without boundary masks
            unet.compile(optimizer=Adam(learning_rate=learning_rate),
                         loss=dice_loss,
                         metrics=[dice_coef, precision, recall])

        unet_hist = unet.fit(train_data_loader[k],
                            epochs=epochs,
                            steps_per_epoch=math.floor(num_train_samples/batch_size),
                            validation_data=val_data_loader[k],
                            validation_steps=math.floor(num_val_samples/batch_size)
                            )
        print(unet_hist.history.keys())
        unet.save('models/unet_lab5_wo_bound_fold{}'.format(k))

        learning_curves(unet_hist, "loss", "val_loss",
                        ["dice_coef", "precision", "recall"],
                        ["val_dice_coef", "val_precision", "val_recall"],
                        save_path='models/unet_lab5_wo_bound_fold{}'.format(k))

        models.append(unet)
        model_histories.append(unet_hist)

    print('finished')
    K.clear_session()

    # import data




