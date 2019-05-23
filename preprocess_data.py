MAX_VALUE = 1
MIN_VALUE = 0

def preprocess_data(data):    # Xtra->Xtra^new
    data = remove_noise(data)
    return data

def remove_noise(data):
    for column_name in data:
        if data[column_name].max >= MAX_VALUE or data[column_name].min <= MIN_VALUE:
            data = data.drop(data[data[column_name] <= MIN_VALUE or data[column_name] >= MAX_VALUE].index)
    return data