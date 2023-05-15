import dataProcessing
import tarfile
import csv
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

extractTar = False

if (extractTar):
    tar_file_path = 'tar/1minute_data_newyork.tar.gz'
    csv_file_path = 'csv/newyork_csv'

    with tarfile.open(tar_file_path, 'r:gz') as tar:
        dataProcessing.tar_to_csv(tar, csv_file_path)

    data27 = dataProcessing.ny_csv_to_sorted_df(csv_file_path, 27, '2019-05-01 00:00:00-05', '2019-05-02 00:00:00-05')
    data27.to_csv('output/data27.csv', index=False)

# dataProcessing.plot_columns_csv('data27.csv',['air1','air2'],'plot')
csv_file_path = 'csv/minute_data.csv'

userData2 = dataProcessing.dgc_csv_to_sorted_df(csv_file_path, 'User 2', '01/01/2023 00:00', '01/01/2024 00:00')
userData2.to_csv('output/user2_data.csv')
# print(userData2)