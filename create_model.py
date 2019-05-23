import pandas as pd
import keras

def loadData(): # Xpaths->Xtra&Xtst
    test_data = pd.read_csv("test.csv")
    training_data = pd.read_csv("training.csv")


def preprocessing():    # Xtra->Xtra^new
    pass

def trainModel():   # Xtra || Xtra^new->model
    pass

def predict():  # model&Xtst
    pass

def writeOutput():  #->submission.csv
    pass
