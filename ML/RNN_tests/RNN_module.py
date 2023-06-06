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

def create_dataset(dataset, window_size = 1): #constructs windows of window_size from the time series data
    data_x, data_y = [], []
    for i in range(len(dataset) - window_size - 1):
        sample = dataset[i:(i + window_size), 0]
        data_x.append(sample) #training values of size window_size
        data_y.append(dataset[i + window_size, 0]) #the true value, which is the value after the window right bound
    return(np.array(data_x), np.array(data_y))

def data_prepare(timeSeries, training_perc, window_size): #takes in a dataframe timeseries data with specified training split percentage
    data_np = timeSeries.values.astype("float32")
    normalizer = MinMaxScaler(feature_range = (0, 1))
    dataset = normalizer.fit_transform(data_np) #apply normalisation on input data
    train_size = int(len(dataset) * training_perc)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    
    print("Splitting training set...")
    print("Number of samples training set: " + str((len(train))))
    print("Number of samples test set: " + str((len(test))))

    train_X, train_Y = create_dataset(train, window_size)
    test_X, test_Y = create_dataset(test, window_size)
    train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
    test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

    print("Preparing training set...")
    print("Shape of training inputs: " + str((train_X.shape)))
    print("Shape of training labels: " + str((train_Y.shape)))
    return train_X, test_X, train_Y, test_Y, dataset, normalizer

def rnn_train(train_X, train_Y, batch_size, epochs, window_size):
    batch_size = 32
    rnn = Sequential()    
    rnn.add(LSTM(16, input_shape = (window_size, 1)))
    rnn.add(Dense(1))
    rnn.compile(loss = "mean_squared_error",  optimizer = "adam", metrics = ['mse'])
    rnn.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size, verbose=1)
    return rnn

def rnn_plot(model, train_X, test_X, train_Y, test_Y, dataset, normalizer):
    #PREDICTIONS
    window_size = train_X.shape[1] #basically the no. of columns = window_size
    len_train = train_X.shape[0]
    train_val_predicted = normalizer.inverse_transform(model.predict(train_X)) #RNN makes predictions here
    train_val_true = normalizer.inverse_transform([train_Y])
    test_val_predicted = normalizer.inverse_transform(model.predict(test_X)) #RNN makes predictions here
    test_val_true = normalizer.inverse_transform([test_Y])
    print(test_val_predicted.shape)
    print("Printing scores of final training and test...")
    print("Training data error: %.2f MSE" % math.sqrt(mean_squared_error(train_val_true[0], train_val_predicted[:, 0])))
    print("Test data error: %.2f MSE" %  math.sqrt(mean_squared_error(test_val_true[0], test_val_predicted[:, 0])))
    
    #PLOTS
    train_plot = np.full_like(dataset, fill_value=np.nan) #need to match dataset size since the start is offset
    train_plot[window_size:len_train + window_size, :] = train_val_predicted #the first prediction is at index window_size, and train_val_predicted to fill that up from there
    test_plot = np.full_like(dataset, fill_value=np.nan)
    test_plot[len_train + (window_size * 2) + 1:len(dataset) - 1, :] = test_val_predicted #fill up offsetted values with the test values 
    
    #matplotlib stuff
    plt.figure(figsize=(15, 5))
    plt.plot(normalizer.inverse_transform(dataset), label="True value")
    plt.plot(train_plot, label="Training predictions")
    plt.plot(test_plot, label="Test predictions")
    plt.xlabel("Time of Day (hrs)")
    plt.ylabel("Consumption")
    plt.title("Comparison true vs. predicted in the training and test set")
    x_ticks = range(0, 1440, 120) #label once every 120 minutes
    x_labels = [f"{i // 60:02d}:00" for i in x_ticks]
    plt.xticks(x_ticks, x_labels)
    plt.legend()
    plt.savefig('test1.png')
    plt.close

def real_data_prepare(timeSeries, window_size): #prepares the input test data in the form of windows
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

def real_rnn_predict_plot(model, test_X, test_Y, dataset, normalizer):
    #PREDICTIONS
    window_size = test_X.shape[1] #basically the no. of columns = window_size
    len_test = test_X.shape[0]
    test_val_predicted = normalizer.inverse_transform(model.predict(test_X)) #RNN makes predictions here
    test_val_true = normalizer.inverse_transform([test_Y])
    print("Printing scores of predictions...")
    print("Test data error: %.2f MSE" %  math.sqrt(mean_squared_error(test_val_true[0], test_val_predicted[:, 0])))
    
    test_plot = np.full_like(dataset, fill_value=np.nan)
    test_plot[window_size:len_test + window_size, :] = test_val_predicted #fill up offsetted values with the test values 
    
    if len_test > 10080: #set label interval in minutes
        label_int = 1440
    elif len_test < 1450:
        label_int = 120
    else:
        label_int = 720
    #matplotlib stuff
    plt.figure(figsize=(15, 5))
    plt.plot(normalizer.inverse_transform(dataset), label="True value")
    plt.plot(test_plot, label="Test predictions")
    plt.xlabel("Time of Day (hrs)")
    plt.ylabel("Consumption")
    plt.title("Comparison true vs. predicted in the training and test set")
    x_ticks = range(0, len_test, label_int) 
    x_labels = [f"{i // 60:02d}:00" for i in x_ticks]
    plt.xticks(x_ticks, x_labels)
    plt.legend()
    plt.savefig('test.png')
    plt.close

def plot_highest_error_period(model, test_X, test_Y, dataset, normalizer, window_size):
    # PREDICTIONS
    test_val_predicted = normalizer.inverse_transform(model.predict(test_X))  # RNN makes predictions here
    test_val_true = normalizer.inverse_transform([test_Y])

    # Calculate prediction errors
    test_errors = np.abs(test_val_true[0] - test_val_predicted[:, 0])

    # Find the 60-minute period with the highest prediction error
    highest_error_index = np.argmax(test_errors)
    highest_error_start = highest_error_index - window_size//2
    highest_error_end = highest_error_index + window_size//2

    print("Printing highest prediction error:")
    print("Error: %.2f" % test_errors[highest_error_index])
    print("Period: Start: %d, End: %d" % (highest_error_start, highest_error_end))

    # Extract the highest error period from the test data
    highest_error_period = dataset[highest_error_start:highest_error_end, :]

    # Create time values for the x-axis
    time_values = range(highest_error_start, highest_error_end)

    # PLOT
    plt.figure(figsize=(10, 5))
    plt.plot(time_values, highest_error_period, label="Predicted values", color='orange')
    plt.plot(time_values, test_val_true[0, highest_error_start:highest_error_end], label="True values", color='blue')
    plt.xlabel("Time")
    plt.ylabel("Consumption")
    plt.title("60-Minute Period with Highest Prediction Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig('highest_error_period.png')
    plt.close
