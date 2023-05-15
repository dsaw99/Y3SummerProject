import tarfile
import csv
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

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

def ny_csv_to_sorted_df(csvfile, dataid, start_time, end_time):
    df = pd.read_csv(csvfile)
    filtered_df = df[df['dataid'] == dataid]  # filter by dataid
    filtered_df['localminute'] = pd.to_datetime(filtered_df['localminute'])  # change localminute to dt object
    sorted_df = filtered_df.sort_values('localminute')  # sort data by dt
    startStamp = pd.to_datetime(start_time)
    endStamp = pd.to_datetime(end_time)
    sorted_df = sorted_df[(sorted_df['localminute'] >= startStamp) & (sorted_df['localminute'] < endStamp)]
    return sorted_df

def dgc_csv_to_sorted_df(csvfile, serial, start_time, end_time):
    df = pd.read_csv(csvfile)
    filtered_df = df[df['Serial Number'] == serial]  # filter by dataid
    filtered_df['DateTime (UTC)'] = pd.to_datetime(filtered_df['DateTime (UTC)'])  # change localminute to dt object
    sorted_df = filtered_df.sort_values('DateTime (UTC)')  # sort data by dt
    startStamp = pd.to_datetime(start_time)
    endStamp = pd.to_datetime(end_time)
    sorted_df = sorted_df[(sorted_df['DateTime (UTC)'] >= startStamp) & (sorted_df['DateTime (UTC)'] < endStamp)]
    return sorted_df

def plot_columns_df(df, column_names, output_filename=None,  dt_name='localminute'):
    x = df[dt_name]
    
    for column_name in column_names:
        y = df[column_name]
        plt.plot(x, y, label=column_name)
    
    plt.xlabel(dt_name)
    plt.title('Plot of Columns against time')
    plt.legend()
    
    if output_filename:
        plt.savefig(output_filename)
        plt.close()
    else:
        plt.show()

def plot_columns_csv(csv_file, column_names, output_filename=None, dt_name='localminute'):
    df = pd.read_csv(csv_file)
    x = df[dt_name]
    
    for column_name in column_names:
        y = df[column_name]
        plt.plot(x, y, label=column_name)
    
    plt.xlabel(dt_name)
    plt.title('Plot of Columns against time')
    plt.legend()
    
    if output_filename:
        plt.savefig(output_filename)
        plt.close()
    else:
        plt.show()

def plot_by_hours_csv(csv_file, start_time, end_time, value, dt_name='DateTime (UTC)'):
    df = pd.read_csv(csv_file)
    df[dt_name] = pd.to_datetime(df[dt_name])
    df['date'] = df[dt_name].dt.date
    start_date = pd.to_datetime(start_time).date()
    end_date = pd.to_datetime(end_time).date()
    selected_df = df[(df['date'] >= start_date) & (df['date'] < end_date)]
    selected_df['time'] = selected_df[dt_name].dt.hour * 60 + selected_df[dt_name].dt.minute

    fig, ax = plt.subplots(figsize=(15, 6))
    for date, group in selected_df.groupby('date'):
        ax.plot(group['time'], group[value], label=date)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel(str(value))
    ax.set_title('Time Series Data for Consecutive Dates')
    ax.legend()
    ax.set_xticks(range(0, 1440, 120))
    ax.set_xticklabels([f"{tick // 60}:00"  for tick in ax.get_xticks()])

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def split_weekdays_weekends(csv_file):
    df = pd.read_csv(csv_file)
    # Convert the 'localminute' column to a datetime object
    df['localminute'] = pd.to_datetime(df['localminute'])
    
    # Extract the day of the week (0 = Monday, 6 = Sunday)
    df['day_of_week'] = df['localminute'].dt.weekday
    
    # Split the DataFrame into two sets based on the day of the week
    weekday_df = df[df['day_of_week'] < 5]  # Weekdays are Monday to Friday (0 to 4)
    weekend_df = df[df['day_of_week'] >= 5]  # Weekends are Saturday and Sunday (5 and 6)

def csv_interpolate(csvfile): #inputs a CSV file and fils up all the missing entries with the average of the past and future entries
    data = pd.read_csv(csvfile)
    data['DateTime (UTC)'] = pd.to_datetime(data['DateTime (UTC)'])
    data.sort_values('DateTime (UTC)', inplace=True)
    # Iterate through the data
    for i in range(len(data) - 1):
        current_timestamp = data['DateTime (UTC)'].iloc[i]
        next_timestamp = data['DateTime (UTC)'].iloc[i + 1]

        if (next_timestamp - current_timestamp).seconds > 60:
            missing_minutes = int((next_timestamp - current_timestamp).seconds / 60) - 1
            interpolation_increment = (data['Avg Wattage'].iloc[i + 1] - data['Avg Wattage'].iloc[i]) / (missing_minutes + 1)

            for j in range(1, missing_minutes + 1):
                interpolated_timestamp = current_timestamp + pd.DateOffset(minutes=j)
                interpolated_value = data['Avg Wattage'].iloc[i] + (interpolation_increment * j)

                # Create a new entry for the missing minute
                new_row = {
                    'Serial Number': 'User 2',
                    'DateTime (UTC)': interpolated_timestamp,
                    'Avg Wattage': interpolated_value
                    # Add other columns as needed
                }

                # Append the new row to the data structure
                data = data.append(new_row, ignore_index=True)

    data.sort_values('DateTime (UTC)', inplace=True)
    data.to_csv('output/interpolated_data.csv', index=False)

def detect_match(csvfile):
    data = pd.read_csv(csvfile)
    occurrence_count = 0
    pattern = [403.5, 196.5, 203.9, 367.8, 1621.4, 1613.8, 1397.9, 937.7, 1023.1, 1110.1, 1181.3, 1121, 1073.1, 1038.8,
               1038.5, 1153.1, 1108.19, 931.3, 869.6, 832.9, 976.1, 934.3]
    
    threshold = 10
    for i in range(len(data)):
        data_minute = [data['Avg Wattage'].iloc[i]]
        correlation = np.correlate(data_minute, pattern)
        if correlation > threshold: 
            occurrence_count += 1
    return occurrence_count

def ny_general_consumption(csvfile):
    df = pd.read_csv(csvfile)
    df['localminute'] = pd.to_datetime(df['localminute'])  # change localminute to dt object
    sorted_df = df.sort_values('localminute')  # sort data by dt
    sum_values = sorted_df.iloc[:, 2:-2].sum(axis=1)

    new_df = pd.DataFrame({
        'dataid': sorted_df['dataid'],
        'localminute': sorted_df['localminute'],
        'overall': sum_values
    })

    new_df.to_csv('output/overall_consumption.csv', index=False)
    return new_df