import SimpleITK as sitk
import os
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

    masks = os.listdir(mask_path)
    masks.sort()

    for m in masks:
        msk = sitk.ReadImage(os.path.join(mask_path, m), sitk.sitkUInt8)
        msk /= msk                                      # normalize mask
        dilated = sitk.BinaryDilate(msk, kernelRadius = (int(2), int(2)))     # compute dilation
        eroded = sitk.BinaryErode(msk, kernelRadius = (int(2), int(2)) )       # compute erosion
        boundary = dilated - eroded                     # compute boundary
        boundary = sitk.Abs(boundary)                   # compute absolute
        sitk.WriteImage(boundary, os.path.join(boundary_path, m))





