from dataloading import  load_streamlines, MyBatchGenerator
from tensorflow.keras.optimizers import Adam
from lstm import get_LSTM_task_2
from analysis import learning_curves
from tensorflow.keras import backend as K
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy

dataPath = '/DL_course_data/Lab5/HCP_lab/'
train_subjects_list = ['613538', '599671', '599469']
# your choice of 3 training subjects


val_subjects_list = ['601127']
# your choice of 1 validation subjects


n_tracts_per_bundle = 20

# Network params
batch_size = 1
look_back = None
n_epochs = 50
dropout_rate = 0.2
learning_rate = 0.0001
n_units = 5

bundles_list = ['CST_left', 'CST_right']
X_train, y_train = load_streamlines(dataPath,
                                    train_subjects_list,
                                    bundles_list,
                                    n_tracts_per_bundle = n_tracts_per_bundle)

X_val, y_val = load_streamlines(dataPath,
                                val_subjects_list,
                                bundles_list,
                                n_tracts_per_bundle = n_tracts_per_bundle)


model = get_LSTM_task_2(batch_size, look_back, n_units, dropout_rate)
model.compile(loss=BinaryCrossentropy(),
              optimizer=Adam(learning_rate=learning_rate),
              metrics=[BinaryAccuracy()])
model.summary()
model_hist = model.fit_generator(MyBatchGenerator(X_train, y_train, batch_size=1),
                    epochs=n_epochs,
                    validation_data=MyBatchGenerator(X_val, y_val, batch_size=1),
                    validation_steps=len(X_val))


learning_curves(model_hist, loss_key='loss',
                validation_loss_key='val_loss',
                metric_keys=['binary_accuracy'],
                validation_metric_keys=['val_binary_accuracy'],
                loss_range=(0, 1), metric_range=(0,1))


K.clear_session()



