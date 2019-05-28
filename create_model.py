import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.svm import SVC


MODEL_TYPES = {
    "NEURAL_NETWORK": 0,
    "SUPPORT_VECTOR_MACHINE": 1,
    "KERNEL_SVM": 2,
    "GAUSSIAN_SVM": 3,
    "RBF_SVM": 4,
}


def create_model(input_shape):
    # model = get_neural_network_model(input_shape) # Model #0
    model = get_support_vector_machine_model()  # Model #1
    # model = get_kernel_SVM_model()    # Model #2
    # model = get_gaussian_SVM_model()  # Model #3
    # model = get_rbf_SVM_model() # Model #4
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


def get_support_vector_machine_model():
    model = SVC(kernel='linear', C=1E10)
    return model


def get_kernel_SVM_model():
    model = SVC(kernel='poly', degree=8, C=1E10)
    return model


def get_gaussian_SVM_model():
    model = SVC(kernel='gaussian', C=1E10)
    return model


def get_rbf_SVM_model():
    model = SVC(kernel='rbf', C=1E10)
    return model
