# file to define network architecture of LSTM
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input, Embedding
from tensorflow.keras.models import Sequential

def get_LSTM(batch_size, look_back, n_units, do_rate=0):
    model = Sequential()
    model.add(LSTM(n_units, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
    model.add(Dropout(do_rate))
    model.add(LSTM(n_units, batch_input_shape=(batch_size, look_back, 1), stateful=True))
    model.add(Dropout(do_rate))
    model.add(LSTM(n_units, batch_input_shape=(batch_size, look_back, 1), stateful=True))
    model.add(Dropout(do_rate))
    model.add(LSTM(n_units, batch_input_shape=(batch_size, look_back, 1), stateful=True))
    model.add(Dropout(do_rate))
    model.add(Dense(1))

    return model