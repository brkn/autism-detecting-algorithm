import keras

def predict_data(model, test_x):  # model&Xtst
    model.predict(test_x, batch_size=None, verbose=0, steps=None, callbacks=None)

