import pandas as pd
import os


def load_data():  # Xpaths->Xtra&Xtst
    current_working_dir = os.getcwd()
    test_path = f"{current_working_dir}/data/test.csv"
    train_path = f"{current_working_dir}/data/train.csv"

    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)
    return test_df, train_df
