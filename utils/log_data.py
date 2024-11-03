import pandas as pd
import os

def log_to_csv(person_name, start_time, end_time, location):
    # Create a DataFrame
    data = {
        "Person Name": [person_name],
        "Start Time": [start_time],
        "End Time": [end_time],
        "Location": [location]
    }
    df = pd.DataFrame(data)

    # Check if the CSV file already exists
    file_exists = os.path.isfile('database.csv')

    # Append or create the CSV file
    if not file_exists:
        df.to_csv('database.csv', header=True, index=False)
    else:
        df.to_csv('database.csv', mode='a', header=False, index=False)
