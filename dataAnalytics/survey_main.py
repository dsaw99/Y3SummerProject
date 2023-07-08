import pandas as pd
import score_mappings

# df = pd.read_csv('csv/survey_results.csv') #parse in CSV as a dataframe
# q1_index = df.columns.get_loc('Q1') #Get the index of column Q1
# survey_data = df.iloc[:, q1_index:] #get all data from Q1 onwards (incl. Q1)
# survey_data = survey_data.drop([0, 1]) #remove the 2 redundant rows below the main column header row.
# survey_data.to_csv('output/survey_clean.csv')


df = pd.read_csv('output/survey_clean.csv')
energy_scores = score_mappings.energy_saving_knowledge(df) #array of energy knowledge score
device_scores = score_mappings.device_knowledge(df) #array of 3 devices scores
overall_scores = score_mappings.overall_knowledge(energy_scores,device_scores) #ranked from 0 to 1
# score_mappings.count_frequency(df)
overall_from_max = score_mappings.distance_from_max(overall_scores, 1) #ranked from 0 to 1