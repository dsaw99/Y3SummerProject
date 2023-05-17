import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('csv/survey_results.csv') #parse in CSV as a dataframe
q1_index = df.columns.get_loc('Q1') 
data = df.iloc[:, q1_index:] #get all data from Q1 onwards (incl. Q1)
data = data.drop([0, 1])['Q5'] #remove the 2 redundant rows below the main column header row.

split_data = data.str.split(',', expand=True).values.flatten() #split up each string 'X,X,X' into individual numbers and make them into an array
counts = pd.Series(split_data).value_counts().sort_index() #count the occurence of each X.

## plot a bar graph
plt.bar(counts.index.astype(int), counts.values)
plt.xlabel('Device Number')
plt.ylabel('Count')
plt.title('Q5: If yes, could you give us your top three?')
plt.show()