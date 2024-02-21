from keras.layers import LSTM, Dense, Embedding,Lambda
from keras.models import Sequential
from keras import backend as K

class LSTM_:
    def __init__(self, i_dim):
        self.i_dim = i_dim

    def build(self):
        self.model = Sequential()
        self.model.add(Lambda(lambda x: K.expand_dims(x, axis=2)))
        self.model.add(LSTM(180, activation="relu"))
        self.model.add(Dense(90))
        self.model.add(Dense(1))
        return self.model
