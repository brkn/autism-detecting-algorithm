import os
import datetime


def submit_to_kaggle(filename):
    printed_filename = filename.split("/")[-1]
    time = datetime.datetime.now()
    time = str(time).split(".")[0]
    submission_command_string = f'kaggle competitions submit -c blg-454e-term-project-competition2019 -f {filename} -m "Submitting {printed_filename} at date {time}"'
    os.system(submission_command_string)
