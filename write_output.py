import os


def write_output(prediction_data):  # ->submission.csv
    latest_file_directory = get_latest_filename()

    with open(latest_file_directory, 'w') as write_file:
        write_file.write("ID,Predicted\n")
        for id, number in enumerate(prediction_data):
            write_file.write(f"{str(id+1)},{number}\n")


def get_latest_filename():
    current_working_dir = os.getcwd()
    submissions_directory = f"{current_working_dir}/submissions"

    files_array = []
    for _, _, files in os.walk(submissions_directory):
        if not files:
            return f"{current_working_dir}/submissions/submission_1.csv"
        for file in files:
            if "submission" in file:
                files_array.append(file.split(".")[0])
    new_sub_number = int(files_array[-1].split("_")[1]) + 1
    return f"{current_working_dir}/submissions/submission_{str(new_sub_number)}.csv"
