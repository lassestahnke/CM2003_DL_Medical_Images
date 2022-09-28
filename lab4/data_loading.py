# script for data loading
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


# following function is adopted from https://www.kaggle.com/code/lqdisme/brain-mri-segmentation-unet-keras
def train_generator(data_frame, batch_size, aug_dict,
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
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        seed=seed)

    # creating instance of ImageDataGenerator for loading and augmenting masks
    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        x_col="mask_list",
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


def load_data(base_path, img_path, target_path, img_size, batch_size, augmenation_dic):
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

    # get training generator for images and masks
    data_gen = train_generator()
    return data_gen


base_path = "/DL_course_data/Lab3/X_ray"
masks = "Mask"
img = "Image"
print(load_data(base_path, img, masks))
