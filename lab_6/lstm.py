# file to define network architecture of LSTM
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input, Embedding, Bidirectional, InputLayer
from tensorflow.keras.models import Sequential

def get_LSTM(batch_size, look_back, n_units, do_rate=0):
    model = Sequential()
    model.add(LSTM(n_units, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
    model.add(Dropout(do_rate))

    model.add(LSTM(n_units, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
    model.add(Dropout(do_rate))

    model.add(LSTM(n_units, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
    model.add(Dropout(do_rate))

    model.add(LSTM(n_units, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=False))
    model.add(Dropout(do_rate))

    model.add(Dense(1))

    return model

def get_LSTM_task_2(batch_size, look_back, n_units, do_rate=0):
    model = Sequential()
    model.add(InputLayer(input_shape=(look_back, 3), batch_size = batch_size))
    model.add(Bidirectional(LSTM(n_units, stateful=True, return_sequences=True)))
    model.add(Dropout(do_rate))

    model.add(LSTM(n_units, stateful=True, return_sequences=True))
    model.add(Dropout(do_rate))

    model.add(LSTM(n_units, stateful=True, return_sequences=True))
    model.add(Dropout(do_rate))

    model.add(LSTM(n_units, stateful=True, return_sequences=False))
    model.add(Dropout(do_rate))

    model.add(Dense(1))

    return model