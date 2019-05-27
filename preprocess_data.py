import pandas as pd

MAX_VALUE = 1
MIN_VALUE = 0


def preprocess_data(data, new_indices=None):    # Xtra->Xtra^new
    data = remove_noise(data)

    if isDataTestData(data):
        data, _ = remove_columns_with_low_variance(data, new_indices)
        return data.values
    else:
        data, new_indices = remove_columns_with_low_variance(data)
        x, y = split_data(data)
        return x.values, y.values, new_indices


def remove_noise(data):
    print(data.shape)

    for column_name in data:
        if data[column_name].max() > MAX_VALUE or data[column_name].min() < MIN_VALUE:
            data = data.drop(
                data[data[column_name] < MIN_VALUE or data[column_name] > MAX_VALUE].index)

    print(data.shape)

    return data


def remove_columns_with_low_variance(data, new_indices=[], threshold=(0.001)):
    from sklearn.feature_selection import VarianceThreshold

    print(data.shape)

    if isDataTestData(data):
        new_indices = new_indices[new_indices != 595]
    else:
        selector = VarianceThreshold(threshold)
        selector.fit(data)
        new_indices = selector.get_support(indices=True)

    new_data = data[data.columns[new_indices]]

    print(new_data.shape)

    return new_data, new_indices


def split_data(data):
    y = data["class"]
    # x = data #To Ömer: Why do we need this?
    x = data.drop("class", axis=1)

    return x, y


def isDataTestData(data):
    isDataTest = not "class" in data
    return isDataTest


def isDataTrainingData(data):
    isDataTraining = "class" in data
    return isDataTraining
