import pandas as pd

def load_data(): # Xpaths->Xtra&Xtst
    test_df = pd.read_csv("data/test.csv")
    train_df = pd.read_csv("data/training.csv")
    return test_df, train_df

