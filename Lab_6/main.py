import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# load csv files:
dataset_train = pd.read_csv('/DL_course_data/Lab5/train_data_stock.csv')
dataset_val = pd.read_csv('/DL_course_data/Lab5/val_data_stock.csv')
# reverse data so that they go from oldest to newest:
dataset_train = dataset_train.iloc[::-1]
dataset_val = dataset_val.iloc[::-1]
# concatenate training and test datasets:
dataset_total = pd.concat((dataset_train['Open'], dataset_val['Open']),
axis=0)
# select the values from the “Open” column as the variables to be predicted:
training_set = dataset_train.iloc[:, 1:2].values
val_set = dataset_val.iloc[:, 1:2].values

# normalize data
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# split training data into T time steps:
T = 60
X_train = []
y_train = []
for i in range(T, len(training_set)):
    X_train.append(training_set_scaled[i-T:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
# normalize the validation set according to the normalization applied to the training set:
inputs = dataset_total[len(dataset_total) - len(dataset_val) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
# split validation data into T time steps:
X_val = []

for i in range(T, T + len(val_set)):
    X_val.append(inputs[i-T:i, 0])
X_val = np.array(X_val)
y_val = sc.transform(val_set)
# reshape to 3D array (format needed by LSTMs -> number of samples, timesteps, input dimension)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))