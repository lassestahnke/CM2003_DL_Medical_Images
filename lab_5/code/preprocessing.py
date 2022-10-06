import SimpleITK as sitk
import os
import numpy as np
def mask_boundaries(path_base, path_mask, path_boundaries):
    """
        Preprocessing function that generates mask boundaries in a seperate folder.
        Performed by dilation, erosion. Boudnaries = dilation - erosion.
        Arguments:
            path_base: base path to mask folder
            path_mask: name of mask folder
            path_boundaries: name of boundary folder
    """

    mask_path = os.path.join(path_base, path_mask)
    boundary_path = os.path.join(path_base, path_boundaries)
    # check if boundary dir exists, if not create it
    if not os.path.exists(boundary_path):
        os.makedirs(boundary_path)

    masks = os.listdir(mask_path)
    masks.sort()

    for m in masks:
        msk = sitk.ReadImage(os.path.join(mask_path, m), imageIO="PNGImageIO")
        vectorRadius = (2, 2)
        kernel = sitk.sitkBall
        dilated = sitk.GrayscaleDilate(msk, vectorRadius, kernel)#, kernelRadius = (int(2), int(2)))     # compute dilation
        eroded = sitk.GrayscaleErode(msk, vectorRadius, kernel)       # compute erosion
        boundary = dilated - eroded                     # compute boundary
        boundary = sitk.Abs(boundary)                   # compute absolute
        boundary_full_path = os.path.join(boundary_path, m)

        sitk.WriteImage(boundary, boundary_full_path, imageIO="PNGImageIO")

    return



