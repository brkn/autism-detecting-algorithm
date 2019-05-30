#!/usr/bin/env python3
from load_data import load_data
from preprocess_data import preprocess_data
from create_model import create_model
from train_model import train_model
from predict_data import predict_data
from write_output import write_output
from submit_latest_file import submit_to_kaggle

MODEL_TYPE = 1  # SUPPORT_VECTOR_MACHINE model


def main():
    test_df, train_df = load_data()

    train_x, train_y, indices_for_masking, pca_instance, scaler_instance = preprocess_data(
        train_df)
    test_x = preprocess_data(test_df, indices_for_masking,
                             pca_instance, scaler_instance)

    model = create_model(train_x.shape[1], MODEL_TYPE)
    train_model(model, MODEL_TYPE, train_x, train_y)
    predictions = predict_data(model, MODEL_TYPE, test_x)

    submission_file_directory = write_output(predictions)
    # submit_to_kaggle(submission_file_directory)


if __name__ == "__main__":
    main()
