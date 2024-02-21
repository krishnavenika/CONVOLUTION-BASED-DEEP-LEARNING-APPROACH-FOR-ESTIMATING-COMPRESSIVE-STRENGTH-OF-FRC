from keras.models import Sequential, Model
from keras.layers import Conv1D, Dense, Flatten, Lambda

from keras import backend as K


class CNN:
    def __init__(self, kernel_size=4, i_dim=19):
       self.kernel_size = kernel_size
       self.i_dim = i_dim

    def build(self):
        # Start building the model
        self.model = Sequential()
        self.model.add(Lambda(lambda x: K.expand_dims(x, axis=2)))
        # First layer Conv2D
        self.model.add(Conv1D(180, input_dim=self.i_dim, kernel_size=self.kernel_size, activation="relu", input_shape=(20, 20)))
        self.model.add(Dense(90, activation="relu"))
        # flatten the data
        self.model.add(Flatten())
        # final layer
        self.model.add(Dense(1))
        return self.model