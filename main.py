import dataProcessing
import tarfile
import csv
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

extractTar = 0
plotDGCData = 1

if (extractTar):
    tar_file_path = 'tar/1minute_data_newyork.tar.gz'
    csv_file_path = 'csv/newyork_csv'

    with tarfile.open(tar_file_path, 'r:gz') as tar:
        dataProcessing.tar_to_csv(tar, csv_file_path)

    data27 = dataProcessing.ny_csv_to_sorted_df(csv_file_path, 27, '2019-05-01 00:00:00-05', '2019-05-02 00:00:00-05')
    data27.to_csv('output/data27.csv', index=False)



if (plotDGCData):
    # import and sort
    csv_file_path = 'csv/minute_data.csv'
    startTime = '2023-04-16 00:00:00'
    endTime = '2023-04-18 00:00:00'
    userData2 = dataProcessing.dgc_csv_to_sorted_df(csv_file_path, 'User 2', startTime, endTime)
    userData2.to_csv('output/user2_data.csv')

    #plot data according to hours
    csv_file_path = 'output/user2_data.csv'
    # dataProcessing.plot_columns_csv('data27.csv',['air1','air2'],'plot')
    # dataProcessing.plot_columns_csv(csv_file_path, ['Avg Wattage'], 'plot', 'DateTime (UTC)')
    dataProcessing.plot_by_hours_csv(csv_file_path, startTime, endTime,'Avg Wattage')