import pandas as pd
import keras

def main():
    test_df, train_df = loadData()
    preprocessing(train_df, test_df)

def loadData(): # Xpaths->Xtra&Xtst
    test_df = pd.read_csv("data/test.csv")
    train_df = pd.read_csv("data/training.csv")
    return test_df, train_df

def preprocessing(train_df, test_df):    # Xtra->Xtra^new
    pass

def trainModel(train_df):   # Xtra || Xtra^new->model
    pass

def predict(test_df):  # model&Xtst
    pass

def writeOutput(result):  #->submission.csv
    pass

if __name__ == "__main__":
    main()
