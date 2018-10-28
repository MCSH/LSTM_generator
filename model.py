import os.path
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import load_model, Sequential

def get_model(Config, maxlen, chars):
    model_name = Config['model_name']
    has_model = os.path.isfile(model_name)

    if has_model:
        print('Load model...')
        return load_model(model_name)
    else:
        print('Build model...')
        LSTM_size = Config['LSTM_size']
        LSTM_count = Config['LSTM_count']
        LSTM_STATEFUL = Config['stateful']
        model = Sequential()
        model.add(LSTM(LSTM_size, return_sequences=True, input_shape=(maxlen, len(chars)), batch_size=Config['batch_size'], stateful=LSTM_STATEFUL))
        model.add(Dropout(0.2))
        for i in range(LSTM_count - 2):
            model.add(LSTM(LSTM_size, return_sequences=True, stateful=LSTM_STATEFUL))
            model.add(Dropout(0.2))
        model.add(LSTM(LSTM_size, return_sequences=False, stateful=LSTM_STATEFUL))
        model.add(Dropout(0.2))
        model.add(Dense(len(chars)))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        return model
