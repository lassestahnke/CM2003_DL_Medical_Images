# script for data loading
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import math


# following function is adopted from https://www.kaggle.com/code/lqdisme/brain-mri-segmentation-unet-keras
def train_generator(data_frame,
                    directory,
                    img_path,
                    target_path,
                    boundary_path,
                    batch_size,
                    aug_dict,
                    image_color_mode="grayscale",
                    mask_color_mode="grayscale",
                    target_size=(256, 256),
                    binary_mask=True,
                    seed=1337):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    # creating instance of ImageDataGenerator for loading and augmenting images
    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        x_col="img_list",
        directory=os.path.join(directory, img_path),
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        seed=seed)

    # creating instance of ImageDataGenerator for loading and augmenting masks
    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        x_col="mask_list",
        directory=os.path.join(directory, target_path),
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        order=0,
        seed=seed)

    if boundary_path is not None:
        boundary_mask_generator = mask_datagen.flow_from_dataframe(
            data_frame,
            x_col="mask_list",
            directory=os.path.join(directory, boundary_path),
            class_mode=None,
            color_mode=mask_color_mode,
            target_size=target_size,
            batch_size=batch_size,
            order=0,
            seed=seed)

    # combining both generators into one
    if boundary_path is not None:
        train_gen = zip(image_generator, boundary_mask_generator, mask_generator)
    else:
        train_gen = zip(image_generator, mask_generator)

    # rescale image and masks and return generator
    if boundary_path is not None:
        for (img, boundary, mask) in train_gen:
            img, boundary, mask = adjust_data(img, boundary, mask, binary_mask)
            yield ([img, boundary], mask)
    else:
        for (img, mask) in train_gen:
            img, mask = adjust_data(img, None, mask, binary_mask)
            yield (img, mask)

    # scaling of grayscale image and discretizing mask values
def adjust_data(img, boundary, mask, binary_mask=True):
    img = img / 255.
    # normalize boundary mask
    if boundary is not None:
        boundary = boundary / 255.
        boundary[boundary > 0] = 1
        boundary[boundary <= 0] = 0

    # Normalize binary segmentation mask
    if binary_mask:
        mask = mask / 255.
        mask[mask > 0] = 1
        mask[mask <= 0] = 0
        if boundary is not None:
            return (img, boundary, mask)
        else:
            return (img, mask)
    else:
        #todo make sure values are rounded after augmentation (can NN interpolation be set?)
        # read pixel of individual labels in mask
        classes = np.unique(mask)
        # every foreground class has its own channel
        shape = (mask.shape[0], mask.shape[1], mask.shape[2], len(classes)-1)
        mask_multiclass = np.zeros(shape) # todo: consider batch size in mask.shape
        for i in range(len(classes)):
            # for multiclass: remove background mask, as it would be 0 anyway
            if i == 0:
                continue
            # set corresponding pixels in channel to 1 if foreground lavel i is present
            mask_class_n = mask * (mask==classes[i]) / classes[i]
            mask_multiclass[:, :, :, i] = mask_class_n[:, :, :, 0]

        if boundary is not None:
            return (img, boundary, mask_multiclass)
        else:
            return (img, mask_multiclass)


def load_data(base_path, img_path, target_path, boundary_path, img_size=(256, 256), val_split=0.0, batch_size=8, augmentation_dic=None,
              binary_mask=True, cross_val=1):
    """
        Function to load data and return a ImageDataGenerator Object for model training

        args:
            base_path: [string] Path to main directory.
            img_path: [string] Name of image directory.
            target_path: [string] Name of mask directory.
            img_size: [tuple] size of image. (img_width, img_height, img_channel).
            val_split: [float] split of data; relative number of validation data; between 0 and 1;
                                only used for cross val = 1
            batch_size: [int] Number of samples per batch.
            augmentation_dic: [dict] Dictionary of augmentation arguments.
                                    See tf.keras.preprocessing.image.ImageDataGenerator doc for info.
            binary_mask: [bool] Bool if binary segmentation mask is used. If True: maks has one channel and two;
                                If False: mask has n-unique channels.
            cross_val: [int] nuber of k-folds for cross validation
        returns:
            data_gen: [iterator]: Training generator
    """

    # get image and mask paths and prepare pd.DataFrame
    if augmentation_dic is None:
        augmentation_dic = {}

    img_list = os.listdir(os.path.join(base_path, img_path))
    img_list.sort()
    target_list = os.listdir(os.path.join(base_path, target_path))
    target_list.sort()

    data = pd.DataFrame(data={'img_list': img_list, 'mask_list': target_list})
    # shuffle dataframe
    data = data.sample(frac=1).reset_index(drop=True)

    # todo combine both Cross Val cases into a more concise code
    if cross_val == 1:
        # todo add option to split data based on number of patients instead of number of images
        # split into train and validation set
        num_train_samples = int(data.shape[0] * (1-val_split))
        num_val_samples = data.shape[0] - num_train_samples
        train_data = data.iloc[:num_train_samples, :]
        val_data = data.iloc[num_train_samples:, :]
        # get training generator for images and masks
        train_data_gen = train_generator(train_data,
                                   directory=base_path,
                                   img_path=img_path,
                                   target_path=target_path,
                                   boundary_path=boundary_path,
                                   batch_size=batch_size,
                                   aug_dict=augmentation_dic,
                                   target_size=img_size,
                                   binary_mask=binary_mask
                                   )

        val_data_gen = train_generator(val_data,
                                   directory=base_path,
                                   img_path=img_path,
                                   target_path=target_path,
                                   boundary_path=boundary_path,
                                   batch_size=batch_size,
                                   aug_dict={},
                                   target_size=img_size,
                                   binary_mask = binary_mask
                                   )
        return [train_data_gen], [val_data_gen], num_train_samples, num_val_samples

    if cross_val > 1:
        chunk_size = math.floor(len(data)/cross_val)
        train_gens = []  # set list of train_generators
        val_gens = []  # set list of validation generators

        num_train_samples = (cross_val-1) * chunk_size
        num_val_samples = chunk_size

        for k in range(cross_val):
            idx = np.arange(len(data))
            mask_train = np.ones(len(idx), dtype=bool)
            mask_train[k*chunk_size:(k+1)*chunk_size] = False
            mask_val = (mask_train == False)

            df_train = data.iloc[idx[mask_train]] # use all data except for k_th fold as training data
            df_val = data.iloc[idx[mask_val]]  # only use k_th fold as validation data

            # get training generator for images and masks
            train_data_gen = train_generator(df_train,
                                             directory=base_path,
                                             img_path=img_path,
                                             target_path=target_path,
                                             boundary_path=boundary_path,
                                             batch_size=batch_size,
                                             aug_dict=augmentation_dic,
                                             target_size=img_size,
                                             binary_mask=binary_mask
                                             )

            val_data_gen = train_generator(df_val,
                                           directory=base_path,
                                           img_path=img_path,
                                           target_path=target_path,
                                           boundary_path=boundary_path,
                                           batch_size=batch_size,
                                           aug_dict={},
                                           target_size=img_size,
                                           binary_mask=binary_mask
                                           )
            train_gens.append(train_data_gen)
            val_gens.append(val_data_gen)
        return train_gens, val_gens, num_train_samples, num_val_samples

