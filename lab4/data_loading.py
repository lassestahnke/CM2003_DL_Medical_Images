# script for data loading
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def load_data(base_path, img_path, target_path):
    img_list = os.listdir(os.path.join(base_path,img_path))
    target_list = os.listdir(os.path.join(base_path,target_path))
    data = pd.DataFrame(data={'x': img_list, 'y': target_list})

    datagen = ImageDataGenerator(rescale=1./255)
    # use flow_from_dataframe
    return data

base_path = "/DL_course_data/Lab3/X_ray"
masks = "Mask"
img = "Image"
print(load_data(base_path, img, masks))



