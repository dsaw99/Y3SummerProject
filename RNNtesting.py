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
import RNNoverall

def data_prepare(timeSeries, window_size): #prepares the input test data in the form of windows
    data_np = timeSeries.values.astype("float32")
    normalizer = MinMaxScaler(feature_range = (0, 1))
    dataset = normalizer.fit_transform(data_np)

    print("Size of dataset for predictions: " + str((len(dataset))))

    data_x, data_y = [], []
    for i in range(len(dataset) - window_size - 1):
        sample = dataset[i:(i + window_size), 0]
        data_x.append(sample) 
        data_y.append(dataset[i + window_size, 0])
    data_x = np.array(data_x)
    test_X = np.reshape(data_x, (data_x.shape[0], data_x.shape[1], 1))
    test_Y = np.array(data_y)
    return test_X, test_Y, dataset, normalizer

def rnn_predict_plot(model, test_X, test_Y, dataset, normalizer):
    #PREDICTIONS
    window_size = test_X.shape[1] #basically the no. of columns = window_size
    len_test = test_X.shape[0]
    test_val_predicted = normalizer.inverse_transform(model.predict(test_X)) #RNN makes predictions here
    test_val_true = normalizer.inverse_transform([test_Y])
    print("Printing scores of predictions...")
    print("Test data error: %.2f MSE" %  math.sqrt(mean_squared_error(test_val_true[0], test_val_predicted[:, 0])))
    
    test_plot = np.full_like(dataset, fill_value=np.nan)
    test_plot[window_size:len_test + window_size, :] = test_val_predicted #fill up offsetted values with the test values 
    
    #matplotlib stuff
    plt.figure(figsize=(15, 5))
    plt.plot(normalizer.inverse_transform(dataset), label="True value")
    plt.plot(test_plot, label="Test predictions")
    plt.xlabel("Time of Day (hrs)")
    plt.ylabel("Consumption")
    plt.title("Comparison true vs. predicted in the training and test set")
    x_ticks = range(0, 1440, 120) #label once every 120 minutes
    x_labels = [f"{i // 60:02d}:00" for i in x_ticks]
    plt.xticks(x_ticks, x_labels)
    plt.legend()
    plt.show()

# #MAIN CODE
# #TRAIN ON OVERALL CONSUMPTION FOR A DAY
# csvfile = 'output/overall_consumption.csv'
# overall_df = pd.read_csv(csvfile)
# overall_df = pd.DataFrame(overall_df.iloc[:, 2]) 
# overall_np = overall_df.values.astype("float32")

# train_X, test_X, train_Y, test_Y, dataset, normalizer = RNNoverall.data_prepare(overall_df, 0.7, 10)
# rnn_model = RNNoverall.rnn_train(train_X, train_Y, batch_size=32, epochs=500, window_size=10)

# #TEST ON OVERALL CONSUMPTION FOR A WHOLE MONTH
# csvfile = 'output\data27_month_overall.csv'
# overall_df = pd.read_csv(csvfile)
# overall_df = pd.DataFrame(overall_df.iloc[:, 2]) 
# overall_np = overall_df.values.astype("float32")

# test_X, test_Y, dataset, normalizer = data_prepare(overall_df, 10)
# rnn_predict_plot(rnn_model, test_X, test_Y, dataset, normalizer)