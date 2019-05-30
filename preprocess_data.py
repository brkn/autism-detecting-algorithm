import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

MAX_VALUE = 1
MIN_VALUE = 0
LABEL_COLUMN_INDICE = 595
NUMBER_OF_BEST_FEATURES = 15
PCA_VARIANCE_PERCENTAGE = 0.90


def preprocess_data(data, new_indices=None, pca_instance=None, scaler_instance=None):    # Xtra->Xtra^new
    data = remove_noise_and_outliers(data)

    if isDataTrainingData(data):
        x, y = split_data(data)
        x, scaler_instance = get_scaled_data(data=x)
        x, pca_instance = get_principal_components(data=x)
        x, new_indices = select_k_best_features(x, y)
        return x.values, y.values, new_indices, pca_instance, scaler_instance

    elif isDataTestData(data):
        data, _ = get_scaled_data(data, scaler_instance)
        data, _ = get_principal_components(data, pca_instance)
        data = apply_mask_to_data(data, new_indices)
        return data.values


def remove_noise_and_outliers(data):
    for column_name in data:
        if data[column_name].max() > MAX_VALUE or data[column_name].min() < MIN_VALUE:
            data = data.drop(
                data[data[column_name] < MIN_VALUE or data[column_name] > MAX_VALUE].index)
    return data


def remove_columns_with_low_variance(data, new_indices=[], threshold=(0.001)):
    # print(data.shape)

    selector = VarianceThreshold(threshold)
    selector.fit(data)
    new_indices = selector.get_support(indices=True)

    new_data = apply_mask_to_data(data, new_indices)

    # print(new_data.shape)
    return new_data, new_indices


def select_k_best_features(train_x, train_y, k=NUMBER_OF_BEST_FEATURES):
    selector = SelectKBest(k=k)
    fitted_selector = selector.fit(train_x, train_y)
    new_indices = fitted_selector.get_support(indices=True)
    new_data = train_x[train_x.columns[new_indices]]
    print_k_best_features(fitted_selector, k)
    # print(new_data.shape)
    return new_data, new_indices


def split_data(data):
    y = data["class"]
    x = data.drop("class", axis=1)
    return x, y


def isDataTestData(data):
    isDataTest = not "class" in data
    return isDataTest


def isDataTrainingData(data):
    isDataTraining = "class" in data
    return isDataTraining


def print_k_best_features(fitted_selector, k):
    dfscores = pd.DataFrame(fitted_selector.scores_)
    dfscores.columns = ['Score']
    print(dfscores.nlargest(k, 'Score'))


def apply_mask_to_data(data, indices):
    if isDataTestData(data):
        indices = indices[indices != LABEL_COLUMN_INDICE]

    masked_data = data[data.columns[indices]]

    return masked_data


def get_principal_components(data, pca_instance=None):
    if pca_instance == None:
        pca_instance = PCA(PCA_VARIANCE_PERCENTAGE)
        pca_instance.fit(data)

    pca_array = pca_instance.transform(data)

    new_columns = ['pca_%i' % i for i in range(pca_array.shape[1])]
    new_data = pd.DataFrame(
        pca_array, index=data.index, columns=new_columns)

    # print(new_data.shape)
    return new_data, pca_instance


def get_scaled_data(data, scaler_instance=None):
    if scaler_instance == None:
        scaler_instance = StandardScaler()
        scaler_instance.fit(data)

    scaled_array = scaler_instance.transform(data)
    new_data = pd.DataFrame(
        scaled_array, index=data.index, columns=data.columns)

    return new_data, scaler_instance
