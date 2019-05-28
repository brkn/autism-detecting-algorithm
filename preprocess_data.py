import numpy as np
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

MAX_VALUE = 1
MIN_VALUE = 0


def preprocess_data(data, new_indices=None):    # Xtra->Xtra^new
    data = remove_noise(data)

    if isDataTrainingData(data):
        data, new_indices = remove_columns_with_low_variance(data)
        x, y = split_data(data)
        x, new_indices = select_k_best_features(x, y)
        return x.values, y.values, new_indices
    else:
        data, _ = remove_columns_with_low_variance(data, new_indices)
        return data.values

def remove_noise(data):
    print(data.shape)

    for column_name in data:
        if data[column_name].max() > MAX_VALUE or data[column_name].min() < MIN_VALUE:
            data = data.drop(
                data[data[column_name] < MIN_VALUE or data[column_name] > MAX_VALUE].index)

    print(data.shape)

    return data


def remove_columns_with_low_variance(data, new_indices=[], threshold=(0.001)):

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


def select_k_best_features(train_x, train_y, k=20):
    selector = SelectKBest(score_func=chi2, k=k)
    fitted_selector = selector.fit(train_x, train_y)
    new_indices = fitted_selector.get_support(indices=True)

    new_data = train_x[train_x.columns[new_indices]]

    print_k_best_features(train_x, fitted_selector, k)

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


def print_k_best_features(train_x, fitted_selector, k):
    dfscores = pd.DataFrame(fitted_selector.scores_)
    dfscores.columns = ['Score']
    print(dfscores.nlargest(k, 'Score'))
