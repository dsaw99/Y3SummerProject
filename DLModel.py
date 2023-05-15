import dataProcessing

# tar_file_path = 'tar/1minute_data_newyork.tar.gz'
csv_file_path = 'csv/newyork_csv'

# Extract the tar.gz file
# with tarfile.open(tar_file_path, 'r:gz') as tar:
#     tar_to_csv(tar, csv_file_path)

# csv to dataframe
data27 = dataProcessing.csv_to_sorted_df(csv_file_path, 27, '2019-05-01 00:00:00-05', '2019-05-02 00:00:00-05')
data27.to_csv('data27.csv', index=False)
# plt.plot(data27['localminute'], data27['air1'])
# plt.xlabel('Time')
# plt.ylabel('air1')
# plt.title('Plot of air1 against localminute')
# plt.show()