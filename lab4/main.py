# CM2003 lab 4
from unet import *
from data_loading import load_data
from tensorflow.keras import metrics

if __name__ == '__main__':
    # set model parameters
    input_size = (128, 128, 1)
    batch_size = 8
    learning_rate = 0.0001
    n_classes = 1
    n_base = 64

    # set paths to data
    base_path = "/DL_course_data/Lab3/X_ray"
    masks = "Mask"
    img = "Image"

    #data_loader = load_data(base_path=base_path,
    #                        img_path=img,
    #                        target_path=masks,
    #                        batch_size=batch_size,
    #                       img_size=input_size)
    # define model
    unet = get_unet(input_shape=input_size, n_classes=n_classes, n_base=n_base, batch_size=batch_size)
    unet.summary()

    # import data




