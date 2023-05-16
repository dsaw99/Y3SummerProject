import RNN_module
import sys
import os

#since dataProcessing is in the parent folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import dataProcessing

import pandas as pd
import numpy as np
import tensorflow as tf
import keras 
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import math
import matplotlib.pyplot as plt
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import io
import requests

run_train, run_ny, run_main = 1, 0, 1
if (run_train):
    if(run_ny):
        csvfile = 'output/overall_consumption.csv'
        value_col = 2
    else:
        csv_file_path = 'csv/minute_data.csv'
        startTime = '2023-04-17 00:00:00'
        endTime = '2023-04-24 00:00:00'
        userData = dataProcessing.dgc_csv_to_sorted_df(csv_file_path, 'User 2', startTime, endTime)
        userData.to_csv('output/user2_april_train.csv')
        dataProcessing.csv_interpolate('output/user2_april_train.csv','output/user2_april_train_interpolated.csv')
        csvfile = 'output/user2_april_train_interpolated.csv'
        value_col = -2
    overall_df = pd.read_csv(csvfile)
    overall_df = pd.DataFrame(overall_df.iloc[:, value_col]) 

    train_X, test_X, train_Y, test_Y, dataset, normalizer = RNN_module.data_prepare(overall_df, 0.7, 10)
    rnn_model = RNN_module.rnn_train(train_X, train_Y, batch_size=32, epochs=500, window_size=10)
    RNN_module.rnn_plot(rnn_model, train_X, test_X, train_Y, test_Y, dataset, normalizer)

if (run_main):
    if(run_ny):
    #TEST ON OVERALL CONSUMPTION FOR A WHOLE MONTH
        csvfile = 'output/data27_month_overall.csv'
        value_col = 2
    else:
        csv_file_path = 'csv/minute_data.csv'
        startTime = '2023-04-24 00:00:00'
        endTime = '2023-05-01 00:00:00'
        userData = dataProcessing.dgc_csv_to_sorted_df(csv_file_path, 'User 2', startTime, endTime)
        userData.to_csv('output/user2_april_test.csv')
        dataProcessing.csv_interpolate('output/user2_april_test.csv','output/user2_april_test_interpolated.csv')
        csvfile = 'output/user2_april_test_interpolated.csv'
        value_col = -2
    overall_df = pd.read_csv(csvfile)
    overall_df = pd.DataFrame(overall_df.iloc[:, value_col]) 
    overall_np = overall_df.values.astype("float32")
    
    test_X, test_Y, dataset, normalizer = RNN_module.real_data_prepare(overall_df, 10)
    RNN_module.real_rnn_predict_plot(rnn_model, test_X, test_Y, dataset, normalizer)    

dgc_csv_sort = 0
if(dgc_csv_sort):
    csv_file_path = 'csv/minute_data.csv'
    startTime = '2023-01-01 00:00:00'
    endTime = '2024-01-01 00:00:00'
    userData2 = dataProcessing.dgc_csv_to_sorted_df(csv_file_path, 'User 2', startTime, endTime)
    userData2.to_csv('output/user2_data.csv')
    dataProcessing.csv_interpolate('output/user2_data.csv','output/interpolated_data.csv')
    dataProcessing.plot_by_hours_csv('output/interpolated_data.csv', '2023-04-16 00:00:00', '2023-04-23 00:00:00', 'Consumption (kWh)', dt_name='DateTime (UTC)')
