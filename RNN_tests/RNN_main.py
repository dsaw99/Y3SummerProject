import RNN_module
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

run_train = 1
if (run_train):
    csvfile = 'output/overall_consumption.csv'
    overall_df = pd.read_csv(csvfile)
    overall_df = pd.DataFrame(overall_df.iloc[:, 2]) 

    train_X, test_X, train_Y, test_Y, dataset, normalizer = RNN_module.data_prepare(overall_df, 0.7, 10)
    rnn_model = RNN_module.rnn_train(train_X, train_Y, batch_size=32, epochs=500, window_size=10)
    RNN_module.rnn_plot(rnn_model, train_X, test_X, train_Y, test_Y, dataset, normalizer)

run_main = 1
if (run_main):
    #TEST ON OVERALL CONSUMPTION FOR A WHOLE MONTH
    csvfile = 'output\data27_month_overall.csv'
    overall_df = pd.read_csv(csvfile)
    overall_df = pd.DataFrame(overall_df.iloc[:, 2]) 
    overall_np = overall_df.values.astype("float32")
    
    test_X, test_Y, dataset, normalizer = RNN_module.real_data_prepare(overall_df, 10)
    RNN_module.real_rnn_predict_plot(rnn_model, test_X, test_Y, dataset, normalizer)    

