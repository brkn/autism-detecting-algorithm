import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

def create_model(input_shape):
    model = Sequential()
    model.add(Dense(100, input_dim=input_shape, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
    # optimizer may be adam
    return model