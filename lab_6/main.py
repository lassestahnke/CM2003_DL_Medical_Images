import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from lstm import get_LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from analysis import learning_curves
import matplotlib.pyplot as plt

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

# model training
batch_size = 16
dropout_rate = 0.2
learning_rate = 0.001
timesteps = T
n_units = 20

model = get_LSTM(batch_size, timesteps, n_units, dropout_rate)
model.compile(loss='mean_squared_error',
              optimizer=Adam(learning_rate=learning_rate),
              metrics=['mean_absolute_error'])
model.summary()
model_hist = model.fit(x=X_train,
                       y=y_train,
                       batch_size=batch_size,
                       validation_data=(X_val, y_val),
                       epochs=100,
          )

learning_curves(model_hist, loss_key='loss',
                validation_loss_key='val_loss',
                metric_keys=['mean_absolute_error'],
                validation_metric_keys=['val_mean_absolute_error'],
                loss_range=(0, 0.05), metric_range=(0,0.2))

predicted_stock_price = model.predict(X_val)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
y_val_scale = sc.inverse_transform(y_val)

plt.plot(y_val_scale,  label="ground truth")
plt.plot(predicted_stock_price, label="prediction")
plt.legend()
plt.show()

K.clear_session()
