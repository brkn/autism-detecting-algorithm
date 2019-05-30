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
C_VALUE_FOR_SVM = 1200


def create_model(input_shape, model_type):
    if model_type == MODEL_TYPES["NEURAL_NETWORK"]:
        model = get_neural_network_model(input_shape)  # Model #0

    elif model_type == MODEL_TYPES["SUPPORT_VECTOR_MACHINE"]:
        model = get_support_vector_machine_model()  # Model #1

    elif model_type == MODEL_TYPES["KERNEL_SVM"]:
        model = get_kernel_SVM_model()    # Model #2

    elif model_type == MODEL_TYPES["GAUSSIAN_SVM"]:
        model = get_gaussian_SVM_model()  # Model #3

    elif model_type == MODEL_TYPES["RBF_SVM"]:
        model = get_rbf_SVM_model()  # Model #4

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
    model = SVC(kernel='linear', C=C_VALUE_FOR_SVM)
    return model


def get_kernel_SVM_model():
    model = SVC(kernel='poly', degree=8, C=C_VALUE_FOR_SVM)
    return model


def get_gaussian_SVM_model():
    model = SVC(kernel='gaussian', C=C_VALUE_FOR_SVM)
    return model


def get_rbf_SVM_model():
    model = SVC(kernel='rbf', C=C_VALUE_FOR_SVM)
    return model
