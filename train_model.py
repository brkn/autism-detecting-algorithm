import numpy as np


def train_model(model, X_train, y_train):   # Xtra || Xtra^new->model
    model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)
