import RNN_module
import sys
import os

#since dataProcessing is in the parent folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import dataProcessing

import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

run_train, run_main, run_anomaly, run_ny = 0, 0, 1, 1
window_size = 60
epochs = 100

if (run_train):
    if(run_ny):
        ##### SUM UP DATA -- TAKES VERY LONG
        # csv_file_path = 'csv/data27_month.csv'
        # start_time = '2019-05-01 00:00:00-05:00'
        # end_time = '2019-05-08 00:00:00-05:00'
        # print('Generating overall consumption....\n')
        # dataProcessing.ny_general_consumption(csv_file_path, 'output/data27_month_overall.csv') #sum up all consumption first, already sorted
        # ny_week = dataProcessing.ny_csv_to_sorted_df('output/data27_month_overall.csv', 27, start_time, end_time)
        # ny_week.to_csv('output/ny_week_train.csv')
        #####
        csvfile = 'output/ny_week_train.csv'
        value_name = 'overall' #just the data column name
    else:
        csv_file_path = 'csv/minute_data.csv'
        startTime = '2023-04-17 00:00:00'
        endTime = '2023-04-24 00:00:00'
        userData = dataProcessing.dgc_csv_to_sorted_df(csv_file_path, 'User 2', startTime, endTime)
        userData.to_csv('output/user2_april_train.csv')
        dataProcessing.csv_interpolate('output/user2_april_train.csv','output/user2_april_train_interpolated.csv')
        csvfile = 'output/user2_april_train_interpolated.csv'
        value_name = 'Avg Wattage' #just the data column name
    overall_df = pd.read_csv(csvfile)
    value_col = overall_df.columns.get_loc(value_name)
    overall_df = pd.DataFrame(overall_df.iloc[:, value_col]) 

    train_X, test_X, train_Y, test_Y, dataset, normalizer = RNN_module.data_prepare(overall_df, 0.7, window_size)
    print('Training model now...\n')
    rnn_model = RNN_module.rnn_train(train_X, train_Y, batch_size=32, epochs=epochs, window_size=window_size,save=True)
    RNN_module.rnn_plot(rnn_model, train_X, test_X, train_Y, test_Y, dataset, normalizer)

if (run_main):
    if(run_ny):
        # #####
        # csv_file_path = 'output/data27_month_overall.csv'
        # start_time = '2019-05-08 00:00:00-05:00'
        # end_time = '2019-05-31 00:00:00-05:00'
        # ny_test = dataProcessing.ny_csv_to_sorted_df('output/data27_month_overall.csv', 27, start_time, end_time)
        # ny_test.to_csv('output/ny_week_test.csv')
        # #####
        rnn_model = load_model("rnn_model.h5")
        csvfile = 'output/ny_week_test.csv'
        value_name = 'overall' #just the data column name
    else:
        csv_file_path = 'csv/minute_data.csv'
        startTime = '2023-04-24 00:00:00'
        endTime = '2023-05-01 00:00:00'
        userData = dataProcessing.dgc_csv_to_sorted_df(csv_file_path, 'User 2', startTime, endTime)
        userData.to_csv('output/user2_april_test.csv')
        dataProcessing.csv_interpolate('output/user2_april_test.csv','output/user2_april_test_interpolated.csv')
        csvfile = 'output/user2_april_test_interpolated.csv'
        value_name = 'Avg Wattage' #just the data column name
    overall_df = pd.read_csv(csvfile)
    value_col = overall_df.columns.get_loc(value_name)
    overall_df = pd.DataFrame(overall_df.iloc[:, value_col]) 
    overall_np = overall_df.values.astype("float32")
    
    test_X, test_Y, dataset, normalizer = RNN_module.real_data_prepare(overall_df, window_size)
    RNN_module.real_rnn_predict_plot(rnn_model, test_X, test_Y, dataset, normalizer)    

if (run_anomaly):
    if(run_ny):
        rnn_model = load_model("rnn_model.h5") #comment out if a new model is used
        csvfile = 'output/ny_week_test.csv' #specify file to contaminate
        num_anomalies = 1000
        anom_amplitude = 8 # 1kW mean for now
        anom_spread = 0.1 # 10W spread for now
        start_time = '2019-05-13 12:00:00-05:00' #contaminate data with anomalies at 12pm, 13 May
        RNN_module.construct_anomalies(csvfile, num_anomalies, anom_amplitude, anom_spread, start_time, 'output/ny_week_test_anomaly.csv')
        value_name = 'overall' #just the data column name #the column at which the overall usage is in
    csvfile_fake = 'output/ny_week_test_anomaly.csv' #data to be used for test
    overall_df = pd.read_csv(csvfile_fake)
    value_col = overall_df.columns.get_loc(value_name)
    overall_df = pd.DataFrame(overall_df.iloc[:, value_col]) 
    overall_np = overall_df.values.astype("float32")

    test_X, test_Y, dataset, normalizer = RNN_module.real_data_prepare(overall_df, window_size)

    # real_data = pd.read_csv(csvfile)['overall'] #original data
    # real_data_df = pd.DataFrame({'overall': real_data})
    # data_np = real_data_df.values.astype("float32")
    # dataset = normalizer.fit_transform(data_np)
    
    RNN_module.real_rnn_predict_plot(rnn_model, test_X, test_Y, dataset, normalizer)    

## UTIL FUNCTION
dgc_csv_sort = 0
if(dgc_csv_sort):
    csv_file_path = 'csv/minute_data.csv'
    startTime = '2023-01-01 00:00:00'
    endTime = '2024-01-01 00:00:00'
    userData2 = dataProcessing.dgc_csv_to_sorted_df(csv_file_path, 'User 2', startTime, endTime)
    userData2.to_csv('output/user2_data.csv')
    dataProcessing.csv_interpolate('output/user2_data.csv','output/interpolated_data.csv')
    dataProcessing.plot_by_hours_csv('output/interpolated_data.csv', '2023-04-16 00:00:00', '2023-04-23 00:00:00', 'Consumption (kWh)', dt_name='DateTime (UTC)')
