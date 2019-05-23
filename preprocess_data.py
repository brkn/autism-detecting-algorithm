import pandas as pd

MAX_VALUE = 1
MIN_VALUE = 0


def preprocess_data(data):    # Xtra->Xtra^new
    data = remove_noise(data)
    data = remove_columns_with_single_unique_value(data)

    if not "class" in data:
        return data.values

    x, y = split_data(data)
    return x.values, y.values


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


def split_data(data):
    y = data["class"]
    x = data
    x = data.drop("class", axis=1)

    return x, y
