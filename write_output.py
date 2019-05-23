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

    numbers = []
    for _, _, files in os.walk(submissions_directory):
        if not files:
            return f"{current_working_dir}/submissions/submission_1.csv"
        for file in files:
            if "submission" in file:
                numbers.append(int(file.split(".")[0].split("_")[1]))
    numbers.sort()
    new_file_number = int(numbers[-1]) + 1
    return f"{current_working_dir}/submissions/submission_{str(new_file_number)}.csv"


write_output("")
