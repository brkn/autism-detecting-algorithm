def train_model(model, X_train, y_train):   # Xtra || Xtra^new->model
    model.fit(X_train, y_train, epochs=50, verbose=0)
