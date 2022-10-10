import pandas as pd
# load csv files:
dataset_train = pd.read_csv(‘/DL_course_data/Lab5/train_data_stock.csv’)
dataset_val = pd.read_csv(‘/DL_course_data/Lab5/val_data_stock.csv’)
# reverse data so that they go from oldest to newest:
dataset_train = dataset_train.iloc[::-1]
dataset_val = dataset_val.iloc[::-1]
# concatenate training and test datasets:
dataset_total = pd.concat((dataset_train[‘Open’], dataset_val[‘Open’]),
axis=0)
# select the values from the “Open” column as the variables to be predicted:
training_set = dataset_train.iloc[:, 1:2].values
val_set = dataset_val.iloc[:, 1:2].values