from keras.models import Sequential
from keras.layers import Dense, Flatten

class SVM:
    def __init__(self, i_dim):
        self.i_dim = i_dim

    def build(self):
        self.model = Sequential()
        self.model.add(Flatten())
        self.model.add(Dense(180, input_dim=self.i_dim, activation='relu'))
        self.model.add(Dense(90, activation='relu'))
        self.model.add(Dense(1))

        return self.model