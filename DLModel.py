import tarfile
import csv
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def tar_to_csv(tar_file, output_csv_filename):
    # Find the CSV file within the tar.gz file
    csv_file = None
    for file in tar_file.getmembers():
        if file.name.endswith('.csv'):
            csv_file = file
            break

    # Read the CSV file if found and save it as a separate CSV file
    if csv_file:
        # Extract the CSV file from the tar.gz file
        extracted_csv = tar_file.extractfile(csv_file)
        csv_reader = csv.reader((line.decode() for line in extracted_csv))  # Decode bytes to strings
        with open(output_csv_filename, 'w', newline='') as output_csv_file:
            csv_writer = csv.writer(output_csv_file)
            for row in csv_reader:
                csv_writer.writerow(row)
        # Close the extracted and output CSV files
        extracted_csv.close()
        output_csv_file.close()
        print(f"CSV file extracted and saved as {output_csv_filename}")
    tar_file.close()

def csv_to_sorted_df(csvfile, dataid, start_time, end_time):
    df = pd.read_csv(csvfile)
    filtered_df = df[df['dataid'] == dataid]  # filter by dataid
    filtered_df['localminute'] = pd.to_datetime(filtered_df['localminute'])  # change localminute to dt object
    sorted_df = filtered_df.sort_values('localminute')  # sort data by dt
    startStamp = pd.to_datetime(start_time)
    endStamp = pd.to_datetime(end_time)
    sorted_df = sorted_df[(sorted_df['localminute'] >= startStamp) & (sorted_df['localminute'] < endStamp)]
    return sorted_df

# tar_file_path = 'tar/1minute_data_newyork.tar.gz'
csv_file_path = 'csv/newyork_csv'

# Extract the tar.gz file
# with tarfile.open(tar_file_path, 'r:gz') as tar:
#     tar_to_csv(tar, csv_file_path)

# csv to dataframe
data27 = csv_to_sorted_df(csv_file_path, 27, '2019-05-01 00:00:00-05', '2019-05-02 00:00:00-05')
plt.plot(data27['localminute'], data27['air1'])
plt.xlabel('Time')
plt.ylabel('air1')
plt.title('Plot of air1 against localminute')
plt.show()