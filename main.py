from load_data import load_data
from preprocess_data import preprocess_data
from create_model import create_model
from train_model import train_model
from predict import predict
from write_output import write_output

def main():
    test_df, train_df = load_data()
    preprocess_data(train_df, test_df)

if __name__ == "__main__":
    main()
