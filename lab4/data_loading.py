# script for data loading
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


# following function is adopted from https://www.kaggle.com/code/lqdisme/brain-mri-segmentation-unet-keras
def train_generator(data_frame,
                    directory,
                    img_path,
                    target_path,
                    batch_size,
                    aug_dict,
                    image_color_mode="grayscale",
                    mask_color_mode="grayscale",
                    target_size=(256, 256),
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
        seed=seed)

    # combining both generators into one
    train_gen = zip(image_generator, mask_generator)

    # rescale image and masks and return generator
    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        yield (img, mask)

    # scaling of grayscale image and discretizing mask values
def adjust_data(img, mask):
    img = img / 255.
    mask = mask / 255.
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return (img, mask)


def load_data(base_path, img_path, target_path, img_size=(256, 256), val_split=0.0, batch_size=8, augmentation_dic=None):
    """
        Function to load data and return a ImageDataGenerator Object for model training

        args:
            base_path: [string] Path to main directory.
            img_path: [string] Name of image directory.
            target_path: [string] Name of mask directory.
            img_size: [tuple] size of image. (img_width, img_height, img_channel).
            batch_size: [int] Number of samples per batch.
            augmentation_dic: [dict] Dictionary of augmentation arguments.
                                    See tf.keras.preprocessing.image.ImageDataGenerator doc for info.
        returns:
            data_gen: [iterator]: Training generator
    """

    # get image and mask paths and prepare pd.DataFrame
    img_list = os.listdir(os.path.join(base_path, img_path))
    target_list = os.listdir(os.path.join(base_path, target_path))
    data = pd.DataFrame(data={'img_list': img_list, 'mask_list': target_list})

    # shuffle dataframe
    data = data.sample(frac=1).reset_index(drop=True)

    # split into train and validation set
    num_train_samples = int(data.shape[0] * (1-val_split))
    num_val_samples = data.shape[0] - num_train_samples

    train_data = data.iloc[:num_train_samples,:]
    val_data = data.iloc[num_train_samples:,:]

    if augmentation_dic is None:
        augmentation_dic={}

    # get training generator for images and masks
    train_data_gen = train_generator(train_data,
                               directory=base_path,
                               img_path=img_path,
                               target_path=target_path,
                               batch_size=batch_size,
                               aug_dict=augmentation_dic,
                               target_size=img_size,
                               )

    val_data_gen = train_generator(val_data,
                               directory=base_path,
                               img_path=img_path,
                               target_path=target_path,
                               batch_size=batch_size,
                               target_size=img_size
                               )
    return train_data_gen, val_data_gen, num_train_samples, num_val_samples