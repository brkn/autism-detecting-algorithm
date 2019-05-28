import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout


def create_model(input_shape):
    # model = get_neural_network_model(input_shape) # This was the first try for a model

    model = get_non_linear_support_vector_machine_model(input_shape)

    return model


def get_neural_network_model(input_shape):
    model = Sequential()
    model.add(Dense(8, input_dim=input_shape, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # optimizer may be adam
    return model


def get_non_linear_support_vector_machine_model(input_shape):
    pass
