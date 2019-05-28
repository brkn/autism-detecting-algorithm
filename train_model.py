import numpy as np  # To Ömer: We don't use this import?
from create_model import MODEL_TYPES


def train_model(model, model_type, X_train, y_train):   # Xtra || Xtra^new->model
    if model_type == MODEL_TYPES["NEURAL_NETWORK"]:
        model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

    elif model_type == MODEL_TYPES["SUPPORT_VECTOR_MACHINE"]:
        model.fit(X_train, y_train)

    elif model_type == MODEL_TYPES["KERNEL_SVM"]:
        model.fit(X_train, y_train)
