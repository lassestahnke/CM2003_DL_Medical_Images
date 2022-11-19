import SimpleITK as sitk
import os
from skimage.io import imread, imsave
from skimage import exposure
import numpy as np


def mask_vessels(path_base, path_segmentation_mask, path_vessel_masks):
    """
        Preprocessing function that generates mask with padding ar in a separate folder.
        Performed by dilation. 
        Arguments:
            path_base: base path to segmentation mask folder
            path_segmentation_mask: name of segmentation mask folder
            path_vessel_masks: name of vessel mask folder
    """

    mask_path = os.path.join(path_base, path_segmentation_mask)
    weight_path = os.path.join(path_base, path_vessel_masks)
    # check if boundary dir exists, if not create it
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    masks = os.listdir(mask_path)
    masks.sort()

    for m in masks:
        msk = sitk.ReadImage(os.path.join(mask_path, m), imageIO="PNGImageIO")
        vectorRadius = (2, 2)
        kernel = sitk.sitkBall
        dilated = sitk.GrayscaleDilate(msk, vectorRadius, kernel)

        weight_full_path = os.path.join(weight_path, m)
        sitk.WriteImage(dilated, weight_full_path, imageIO="PNGImageIO")

    return


def rescale_intensity_from_directory(base_dir, img_dir, mask_dir, new_base_dir=None, new_img_dir=None,
                                     new_mask_dir=None, extension=".png"):
    """
    Load images and segmentation masks to grayscale. Change names of files
    so that names of images and masks are coherent and store them in img/mask directory within a train
    directory

    args:
        base_dir: [string] path to base directory of drive dataset
        img_dir: [string] name of image directory
        mask_dir: [string] name of mask directory; If mask_dir is None, no masks are processed
        new_base_dir: [string] path to new dataset folder; if new_base_dir is None, old dir is overwritten
        extension: [string] extension of new image and mask files
    return
    """

    # set new directory names:
    if new_base_dir is None:
        new_base_dir = base_dir

    if new_img_dir is None:
        new_img_dir = img_dir

    # check if new directories exist, if not then create
    if not os.path.isdir(os.path.join(new_base_dir)):
        os.mkdir(os.path.join(new_base_dir))

    if not os.path.isdir(os.path.join(new_base_dir, new_img_dir)):
        os.mkdir(os.path.join(new_base_dir, new_img_dir))

    if mask_dir is not None:
        if new_mask_dir is None:
            new_mask_dir = mask_dir

        if not os.path.isdir(os.path.join(new_base_dir, new_mask_dir)):
            os.mkdir(os.path.join(new_base_dir, new_mask_dir))

    # load and process images:
    for file in os.listdir(os.path.join(base_dir, img_dir)):
        image = imread(os.path.join(base_dir, img_dir, file), as_gray=True)
        # use contrast stretching
        p1, p99 = np.percentile(image, (1, 99))
        image = exposure.rescale_intensity(image, in_range=(p1, p99))
        imsave(os.path.join(new_base_dir, new_img_dir, file[0:-4] + extension), image)

    # load and process masks:
    if mask_dir is not None:
        for file in os.listdir(os.path.join(base_dir, mask_dir)):
            mask = imread(os.path.join(base_dir, mask_dir, file), as_gray=True)
            imsave(os.path.join(new_base_dir, new_mask_dir, file[0:-4] + extension), mask)
