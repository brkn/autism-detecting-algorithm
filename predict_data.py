from create_model import MODEL_TYPES


def predict_data(model, model_type, test_x):  # model&Xtst
    if model_type == MODEL_TYPES["NEURAL_NETWORK"]:
        predictions = model.predict(
            test_x, batch_size=None, verbose=0, steps=None)
        predictions_rounded = []
        for prediction in predictions:
            if prediction >= 0.5:
                predictions_rounded.append(1)
            else:
                predictions_rounded.append(0)

    elif model_type == MODEL_TYPES["SUPPORT_VECTOR_MACHINE"]:
        predictions_rounded = model.predict(test_x)

    elif model_type == MODEL_TYPES["KERNEL_SVM"]:
        predictions_rounded = model.predict(test_x)

    elif model_type == MODEL_TYPES["GAUSSIAN_SVM"]:
        predictions_rounded = model.predict(test_x)

    return predictions_rounded
