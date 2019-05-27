def predict_data(model, test_x):  # model&Xtst
    predictions = model.predict(test_x, batch_size=None, verbose=0, steps=None)
    predictions_rounded = []
    for prediction in predictions:
        if prediction >= 0.5:
            predictions_rounded.append(1)
        else:
            predictions_rounded.append(0)
    return predictions_rounded
