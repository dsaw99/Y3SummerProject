import tarfile
import csv
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import os
from decimal import Decimal
from decimal import Decimal
from collections import defaultdict

class Dataset:
    def __init__(self, df):
        if isinstance(df, str):
            if df.endswith('.tar'):
                self._load_data_from_tar(df)
            elif df.endswith('.csv'):
                self._load_data_from_csv(df)
            else:
                raise ValueError("Unsupported file format. Only .tar and .csv are supported.")
        elif isinstance(df, pd.DataFrame):
            self.df = df
        else:
            raise ValueError("Invalid input type. Expected str (file path) or pd.DataFrame.")

    # PRIVATE

    def _GetHourlyConsumpt24Cols(self):   # Each Column corrsponds to 1 hour (from 0-23) and each row is a day.
        if hasattr(self.df, "localminute"):
            self.df['localminute'] = pd.to_datetime(self.df['localminute'])
            self.df['hour'] = self.df['localminute'].dt.hour
            self.df['date'] = self.df['localminute'].dt.date

            hourly_consumption = self.df.groupby(['date', 'hour', 'dataid'])['overall'].sum()
            result = hourly_consumption.unstack(level=[1, 2])
            result = result.fillna(0)

        else:
            self.df = self.df.drop(['Serial Number', 'Device ID', 'Device Type', 'Device Make', 'Device Model', 'Device Location', 'User Device Name', 'Production (kWh)', 'Avg Wattage'], axis=1)
            self.df = self.df.loc[self.df['Sense Device Name'] == 'Consumption']
            self.df = self.df.drop('Sense Device Name', axis=1)
            self.df = self.df.rename(columns={"DateTime (UTC)": "localminute"})
            self.df = self.df.rename(columns={"Consumption (kWh)": "overall"})

            self.df['localminute'] = pd.to_datetime(self.df['localminute'], dayfirst=True)
            self.df['hour'] = self.df['localminute'].dt.hour
            self.df['date'] = self.df['localminute'].dt.date

            hourly_consumption = self.df.groupby(['date', 'hour'])['overall'].sum()
            result = hourly_consumption.unstack(level=1)
            result = result.fillna(0)

        result.to_csv('output/column_data.csv', index=False)
        return result

    def _load_data_from_tar(self, file_path):
        with tarfile.open(file_path, 'r') as tar:
            file_name = tar.getnames()[0]
            file_content = tar.extractfile(file_name).read()
            self.df = pd.read_csv(file_content)

    def _load_data_from_csv(self, file_path):
        self.df = pd.read_csv(file_path, low_memory=False)

    #PUBLIC
    
    def get_hourly_consumption(self):
        self.df['localminute'] = pd.to_datetime(self.df['localminute'])
        self.df.set_index('localminute', inplace=True)
        hourly_df = self.df.resample('H').sum()
        hourly_df = pd.concat([self.df, self.df.index.to_frame()], axis=1).reset_index(drop=True)
        return hourly_df

    def split_weekdays_weekends(self):
        if hasattr(self.df, "localminute"):
            self.df['localminute'] = pd.to_datetime(self.df['localminute'])
            self.df['day_of_week'] = self.df['localminute'].dt.weekday
        else:
            self.df['DateTime (UTC)'] = pd.to_datetime(self.df['DateTime (UTC)'], dayfirst=True)
            self.df['day_of_week'] = self.df['DateTime (UTC)'].dt.weekday
        
        weekday_df = self.df[self.df['day_of_week'] < 5].copy()
        weekend_df = self.df[self.df['day_of_week'] >= 5].copy()
        
        weekday_df.drop('day_of_week', axis=1, inplace=True)
        weekend_df.drop('day_of_week', axis=1, inplace=True)
        
        return weekday_df, weekend_df

    def csv_interpolate(self):
        data = self.df
        data['DateTime (UTC)'] = pd.to_datetime(data['DateTime (UTC)'])
        data.sort_values('DateTime (UTC)', inplace=True)

        for i in range(len(data) - 1):
            current_timestamp = data['DateTime (UTC)'].iloc[i]
            next_timestamp = data['DateTime (UTC)'].iloc[i + 1]

            if (next_timestamp - current_timestamp).seconds > 60:
                missing_minutes = int((next_timestamp - current_timestamp).seconds / 60) - 1
                interpolation_increment = (data['Avg Wattage'].iloc[i + 1] - data['Avg Wattage'].iloc[i]) / (missing_minutes + 1)

                for j in range(1, missing_minutes + 1):
                    interpolated_timestamp = current_timestamp + pd.DateOffset(minutes=j)
                    interpolated_value = data['Avg Wattage'].iloc[i] + (interpolation_increment * j)

                    new_row = {
                        'Serial Number': 'User 2',
                        'DateTime (UTC)': interpolated_timestamp,
                        'Avg Wattage': interpolated_value
                    }

                    data = data.append(new_row, ignore_index=True)

        data.sort_values('DateTime (UTC)', inplace=True)
        data.to_csv('output/interpolated_data.csv', index=False)

    def detect_match(self):
        data = self.df
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

    def ny_general_consumption(self):
        self.df['localminute'] = pd.to_datetime(self.df['localminute'])
        sorted_df = self.df.sort_values('localminute')
        sum_values = sorted_df.iloc[:, 2:-2].sum(axis=1)

        new_df = pd.DataFrame({
            'dataid': sorted_df['dataid'],
            'localminute': sorted_df['localminute'],
            'overall': sum_values
        })

        new_df.to_csv('output/overall_consumption.csv', index=False)
        return new_df

    # PLOTTING METHODS

    def plot_hourly_boxplot(self, plotName="boxplot"):
        result = self._GetHourlyConsumpt24Cols()
        data = [result[column].values for column in result.columns]
        
        fig, ax = plt.subplots(figsize=(10, 6)) 

        custom_colors = ["#0066cc"] * len(result.columns)

        sns.boxplot(data=data, ax=ax, showfliers=False, palette=custom_colors)
        ax.set_title('Box Plot for Each Hour')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Value')
        ax.set_xticklabels(result.columns, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(plotName, dpi=300)
        plt.close()


    def plot_hourly_linegraph(self, plotName="linegraph"):
        result = self._GetHourlyConsumpt24Cols()
        data = [result[column].values for column in result.columns]
        
        fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as per your preference

        # Line graph for median, Q1, and Q3
        n = len(data)
        positions = np.arange(1, n + 1)
        medians = [np.median(d) for d in data]
        q1s = [np.percentile(d, 25) for d in data]
        q3s = [np.percentile(d, 75) for d in data]

        upper_outliers = [np.max(d[d > q3]) if np.any(d > q3) else None for d, q3 in zip(data, q3s)]
        lower_outliers = [np.min(d[d < q1]) if np.any(d < q1) else None for d, q1 in zip(data, q1s)]

        # Plot median, Q1, Q3, upper outliers (U), and lower outliers (L)
        ax.plot(positions, medians, marker='o', color='red', label='Median', linewidth=1, markersize=5)
        ax.plot(positions, q1s, marker='.', color='green', label='Q1', linestyle='dashed', linewidth=0.75, markersize=3)
        ax.plot(positions, q3s, marker='.', color='blue', label='Q3', linestyle='dashed', linewidth=0.75, markersize=3)

        # Plot upper outliers (U) as a line graph
        upper_outliers_valid = [o for o in upper_outliers if o is not None]
        if len(upper_outliers_valid) > 0:
            ax.plot(positions, [q3 if o is None else o for q3, o in zip(q3s, upper_outliers)], color='orange',
                    linestyle='dashed', linewidth=0.5)
            ax.scatter(positions, upper_outliers, marker='^', color='orange', label='Upper Outlier', s=15)

        # Plot lower outliers (L) as a line graph
        lower_outliers_valid = [o for o in lower_outliers if o is not None]
        if len(lower_outliers_valid) > 0:
            ax.plot(positions, [q1 if o is None else o for q1, o in zip(q1s, lower_outliers)], color='purple',
                    linestyle='dashed', linewidth=0.5)
            ax.scatter(positions, lower_outliers, marker='v', color='purple', label='Lower Outlier', s=15)

        ax.set_title('Line Graph for Median, Q1, Q3, Upper Outliers, and Lower Outliers')
        ax.set_xlabel('Group')
        ax.set_ylabel('Value')
        ax.set_xticks(positions)
        ax.set_xticklabels(result.columns, rotation=45, ha='right')
        ax.legend()

        plt.tight_layout()
        plt.savefig(plotName, dpi=300)  # Increase dpi for higher resolution if needed
        plt.close()


    def plot_columns(self, column_names, output_filename=None, dt_name='localminute'):
        x = self.df[dt_name]
        fig, ax = plt.subplots()

        for column_name in column_names:
            y = self.df[column_name]
            ax.plot(x, y, label=column_name)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

        plt.xticks(rotation=10)

        plt.xlabel(dt_name)
        plt.title('Plot of Columns against time')
        plt.legend()
        
        if output_filename:
            plt.savefig(output_filename)
            plt.close()
        else:
            plt.show()

    def plot_by_hours(self, start_time, end_time, value, dt_name='DateTime (UTC)'):
        self.df[dt_name] = pd.to_datetime(self.df[dt_name])
        self.df['date'] = self.df[dt_name].dt.date
        start_date = pd.to_datetime(start_time).date()
        end_date = pd.to_datetime(end_time).date()
        selected_df = self.df[(self.df['date'] >= start_date) & (self.df['date'] < end_date)]
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

    ''' def getDailyAverage(self):
        df = self.df[self.df['Sense Device Name'] == 'Consumption'].copy()
        df['DateTime (UTC)'] = pd.to_datetime(df['DateTime (UTC)'], dayfirst=True)
        df['Date'] = df['DateTime (UTC)'].dt.date
        daily_average = df.groupby('Date')['Consumption (kWh)'].sum()
        average_consumption = daily_average.mean()
        average_consumption = round(average_consumption, 2)

        # Convert the daily average to annual consumption
        annual_consumption = average_consumption * 365

        # Determine the household size based on the annual consumption
        if annual_consumption <= (1800 + 2900) / 2:
            household_size = "1-2 people"
        elif (1800 + 2900) / 2 < annual_consumption <= (2900 + 4300) / 2:
            household_size = "2-3 people"
        else:
            household_size = "4+ people"

        return {"avg": average_consumption, "householdSize": household_size} '''

    def getDailyAverageInterval(self, last_day, first_day=0):
        df = self.df[self.df['Sense Device Name'] == 'Consumption'].copy()
        df['DateTime (UTC)'] = pd.to_datetime(df['DateTime (UTC)'], dayfirst=True)
        df.set_index('DateTime (UTC)', inplace=True)  # Set DateTime as index for resampling

        # Resample to daily frequency and sum up the minute data
        df = df.resample('D').sum().reset_index()

        # Calculate the date range
        max_date = df['DateTime (UTC)'].max().date()
        start_date = max_date - pd.Timedelta(days=first_day)
        end_date = max_date - pd.Timedelta(days=last_day)

        # Select the rows within the date range
        df = df[(df['DateTime (UTC)'].dt.date >= end_date) & (df['DateTime (UTC)'].dt.date < start_date)]

        daily_average = df['Consumption (kWh)'].mean()
        average_consumption = round(daily_average, 2)

        # Convert the daily average to annual consumption
        annual_consumption = average_consumption * 365

        # Determine the household size based on the annual consumption
        if annual_consumption <= (1800 + 2900) / 2:
            household_size = "1-2 people"
        elif (1800 + 2900) / 2 < annual_consumption <= (2900 + 4300) / 2:
            household_size = "2-3 people"
        else:
            household_size = "4+ people"

        return {"avg": average_consumption, "householdSize": household_size}

    '''def getScore(self):
        avg = self.getDailyAverage()["avg"]

        directory = "output/"
        total_avg = 0
        file_count = 0

        for filename in os.listdir(directory):
            if filename.startswith("User") and filename.endswith(".csv"):
                file_path = os.path.join(directory, filename)
                data = Dataset(file_path)
                total_avg += data.getDailyAverage()["avg"]
                file_count += 1

        # Average across all users
        if file_count != 0:
            total_avg /= file_count
        else:
            total_avg = 0

        # Prevent division by zero
        if avg != 0:
            score = (total_avg / avg) * 40
        else:
            score = 0

        score = max(0, min(score, 100))

        return int(score)'''

    def get_consumption_last_7_days(self):
        df = self.df[self.df['Sense Device Name'] == 'Consumption'].copy()
        df['DateTime (UTC)'] = pd.to_datetime(df['DateTime (UTC)'], dayfirst=True) # Make sure dayfirst is True as date is in DD/MM/YYYY format
        df.set_index('DateTime (UTC)', inplace=True)  # Set 'DateTime (UTC)' as the index

        # Resample to daily frequency and sum up the minute data
        daily_data = df.resample('D').sum()

        # Get last 7 days of data
        last_7_days = daily_data.last('7D')

        # Return the data as a list of dictionaries
        return {date.strftime('%Y-%m-%d'): consumption for date, consumption in last_7_days['Consumption (kWh)'].items()}
    

    def get_consumption_last_7_days2(self):
        # Parse date
        self.df['DateTime (UTC)'] = pd.to_datetime(self.df['DateTime (UTC)'], dayfirst=True)
        self.df.set_index('DateTime (UTC)', inplace=True)  # Set 'DateTime (UTC)' as the index

        # Group by day and device type and sum consumption
        daily_data = self.df.groupby([pd.Grouper(freq='D'), 'Device Type']).sum().reset_index()

        # Get the most recent date in your data
        most_recent_date = daily_data['DateTime (UTC)'].max()

        # Get last 7 days of data
        last_7_days = daily_data[daily_data['DateTime (UTC)'] >= most_recent_date - pd.DateOffset(days=6)]

        # Prepare the data in the format required for the D3.js stacked bar chart
        result = {}
        for _, row in last_7_days.iterrows():
            date = row['DateTime (UTC)'].strftime('%Y-%m-%d')
            device_type = row['Device Type']
            consumption = Decimal(row['Consumption (kWh)']).quantize(Decimal('0.000'))  # Round consumption to 3 decimal places

            if date not in result:
                result[date] = {}

            result[date][device_type] = consumption

        # Create a complete set of dates for the past 7 days
        complete_date_range = pd.date_range(end=most_recent_date, periods=7).format(formatter=lambda x: x.strftime('%Y-%m-%d'))

        # Fill in any missing dates in the result
        for date in complete_date_range:
            if date not in result:
                result[date] = {}

        return result
    
    def get_hourly_bar_chart(self, date, consumption_type):
        self.df['DateTime (UTC)'] = pd.to_datetime(self.df['DateTime (UTC)'], format='%d/%m/%Y %H:%M')
        self.df['Hour'] = self.df['DateTime (UTC)'].dt.hour
        # Format the date string to match the format in your dataset
        formatted_date = pd.to_datetime(date, format='%Y-%m-%d').strftime('%d/%m/%Y')

        if(consumption_type == 'Consumption'):
            filtered_data = self.df[
                (self.df['DateTime (UTC)'].dt.strftime('%d/%m/%Y') == formatted_date) &
                (self.df['Sense Device Name'] == consumption_type)
            ]  
        else:
            filtered_data = self.df[
                (self.df['DateTime (UTC)'].dt.strftime('%d/%m/%Y') == formatted_date) &
                (self.df['Device Type'] == consumption_type)
            ]
            
        hourly_data = filtered_data.groupby('Hour')['Consumption (kWh)'].sum().reset_index()

        # Ensure that the 'Hour' column is complete, by reindexing to the full hour range (0 to 23)
        hourly_data.set_index('Hour', inplace=True)
        hourly_data = hourly_data.reindex(range(0, 24), fill_value=0).reset_index()

        data_for_chart = hourly_data.to_dict('records')

        return data_for_chart
    
    def getWeeklyRatio(self):
        avgWeek = self.getDailyAverageInterval(7)["avg"]
        avgLastWeek = self.getDailyAverageInterval(14, 7)["avg"]
        ratio = avgWeek / avgLastWeek

        if ratio > 1:
            percentage_change = (ratio - 1) * 100
            return f"Your consumption was {percentage_change:.2f}% higher than last week."
        else:
            percentage_change = (1 - ratio) * 100
            return f"Your consumption was {percentage_change:.2f}% lower than last week."
        
    def get_average_consumption_last_x_days(self, x=6):
        # Parse date
        self.df['DateTime (UTC)'] = pd.to_datetime(self.df['DateTime (UTC)'], dayfirst=True)
        self.df.set_index('DateTime (UTC)', inplace=True)  # Set 'DateTime (UTC)' as the index

        # Create new column that takes 'Device Type' if it exists and is not null, else 'Sense Device Name'
        self.df['Device'] = self.df['Device Type'].where(self.df['Device Type'].notna(), self.df['Sense Device Name'])

        daily_data = self.df.groupby([pd.Grouper(freq='D'), 'Device']).sum().reset_index()

        second_most_recent_date = daily_data['DateTime (UTC)'].unique()[-2]

        last_x_days = daily_data[(daily_data['DateTime (UTC)'] >= second_most_recent_date - pd.DateOffset(days=x)) & (daily_data['DateTime (UTC)'] <= second_most_recent_date)]

        # Group by device type and calculate average consumption over last x days
        avg_consumption = last_x_days.groupby('Device')['Consumption (kWh)'].mean()
        result = avg_consumption.round(3).to_dict()

        return result
    
    def getRelevancy(self, DeviceType):
        averages = self.get_average_consumption_last_x_days()
        result = averages[DeviceType] / averages['Consumption'] * 100
        result = round(result, 2)
        return result


def get_overall_average_consumption(directory, interval=6):
    file_averages = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(directory, filename))
            ec = Dataset(df)
            file_average = ec.get_average_consumption_last_x_days(interval)
            file_averages.append(file_average)

    all_devices = set().union(*file_averages)

    overall_average = {}
    for device in all_devices:
        device_values = [file_average[device] for file_average in file_averages if device in file_average]
        if device_values:
            overall_average[device] = round(sum(device_values) / len(device_values), 3)

    return overall_average