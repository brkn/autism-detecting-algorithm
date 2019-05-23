#!/usr/bin/env python3
from load_data import load_data
from preprocess_data import preprocess_data
from create_model import create_model
from train_model import train_model
from predict_data import predict_data
from write_output import write_output
from submit_latest_file import submit_to_kaggle


def main():
    test_df, train_df = load_data()
    train_x, train_y = preprocess_data(train_df)
    test_x = preprocess_data(test_df)

    model = create_model(train_x.shape[1])
    train_model(model, train_x, train_y)
    predictions = predict_data(model, test_x)

    submission_file_directory = write_output(predictions)

    submit_to_kaggle(submission_file_directory)


if __name__ == "__main__":
    main()
