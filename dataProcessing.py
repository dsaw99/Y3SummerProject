import tarfile
import csv
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
    
    return weekday_df, weekend_df