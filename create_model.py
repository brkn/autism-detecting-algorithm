import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense

def create_model(input_shape):
    model = Sequential()
    model.add(Dense(4, input_dim=input_shape, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','loss'])
    return model