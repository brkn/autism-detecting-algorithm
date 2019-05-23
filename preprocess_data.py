import pandas as pd

MAX_VALUE = 1
MIN_VALUE = 0


def preprocess_data(data):    # Xtra->Xtra^new
    data = remove_noise(data)
    data = remove_columns_with_single_unique_value(data)
    return data


def remove_noise(data):
    for column_name in data:
        if data[column_name].max() > MAX_VALUE or data[column_name].min() < MIN_VALUE:
            data = data.drop(
                data[data[column_name] < MIN_VALUE or data[column_name] > MAX_VALUE].index)
    return data


def remove_columns_with_single_unique_value(data):
    nunique_datas = data.apply(pd.Series.nunique)
    cols_to_drop = nunique_datas[nunique_datas == 1].index
    data.drop(cols_to_drop, axis=1)
    return data