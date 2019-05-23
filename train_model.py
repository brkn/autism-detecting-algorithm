import numpy as np

def train_model(model, X_train, y_train):   # Xtra || Xtra^new->model
    model.fit(np.array(X_train), np.array(y_train), epochs=50, verbose=0)
