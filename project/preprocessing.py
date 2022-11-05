import SimpleITK as sitk
import os

def mask_vessels(path_base, path_segmentation_mask, path_vessel_masks):
    """
        Preprocessing function that generates mask with padding ar in a seperate folder.
        Performed by dilation. 
        Arguments:
            path_base: base path to segmentation mask folder
            path_segmentation_mask: name of segmentation mask folder
            path_vessel_masks: name of vessel mask folder
    """

    mask_path = os.path.join(path_base, path_segmentation_mask)
    boundary_path = os.path.join(path_base, path_vessel_masks)
    # check if boundary dir exists, if not create it
    if not os.path.exists(boundary_path):
        os.makedirs(boundary_path)

    masks = os.listdir(mask_path)
    masks.sort()

    for m in masks:
        msk = sitk.ReadImage(os.path.join(mask_path, m), imageIO="PNGImageIO")
        vectorRadius = (2, 2)
        kernel = sitk.sitkBall
        dilated = sitk.GrayscaleDilate(msk, vectorRadius, kernel)

        boundary_full_path = os.path.join(boundary_path, m)
        sitk.WriteImage(dilated, boundary_full_path, imageIO="PNGImageIO")

    return