import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

## Parse in Yuhe's Device consumption list, taking only the required columns (consumption per month) ##
device_df = pd.read_csv('csv/1000users_consumption_comparison.csv') #parse in CSV as a dataframe
# print(df.head(5))
device_df = device_df[['User_ID',
                       'Washing Machine Consumption (kWh)',
                       'Dishwasher Consumption (kWh)',
                       'Dryer Consumption (kWh)',
                       'Fridge Consumption (kWh)',
                       'Washing Machine Change?',
                       'New Washing Machine Consumption (kWh)',
                       'Dryer Change?',
                       'New Dryer Consumption (kWh)',
                       'Fridge Change?',
                       'New Fridge Consumption (kWh)']]

device_df = device_df.replace(' ', 0)

## From Yuhe's Device consumption list, generate random user profile AND random always on consumption levels ##
print('Generating user profiles and always on monthly consumpiton ...\n')
profiles = ['Tech-savvy', 'Traditional', 'Energy Champion', 'Laid-back']
counts = [98, 798, 98, 6]
user_profiles = np.repeat(profiles, counts)
user_profiles = np.random.permutation(user_profiles)[:1000]  # Shuffle the profiles
device_df.insert(1, 'Profile', user_profiles)

for profile,count in zip(profiles,device_df['Profile'].value_counts()):
    print(profile, ':', count)
print()

always_on_mean = 150
always_on_sd = 20

for month in range(12):
    device_df[f'Month{month+1}'] = np.random.normal(always_on_mean, always_on_sd, size=1000) #label month1 to month12 for always on
    monthly_df = device_df

## Consumption totals ##
monthly_df['Annual Consumption (kWh)'] = 12 * (monthly_df['Washing Machine Consumption (kWh)'] +
                                                monthly_df['Dryer Consumption (kWh)'] +
                                                monthly_df['Fridge Consumption (kWh)'] +
                                                monthly_df['Dishwasher Consumption (kWh)']) + \
                                          monthly_df['Month1'] + monthly_df['Month2'] + monthly_df['Month3'] + \
                                          monthly_df['Month4'] + monthly_df['Month5'] + monthly_df['Month6'] + \
                                          monthly_df['Month7'] + monthly_df['Month8'] + monthly_df['Month9'] + \
                                          monthly_df['Month10'] + monthly_df['Month11'] + monthly_df['Month12']

max_overall = round(monthly_df['Annual Consumption (kWh)'].max(),1)
min_overall = round(monthly_df['Annual Consumption (kWh)'].min(),1) 
avg_overall = round(monthly_df['Annual Consumption (kWh)'].mean(),1) 
print(f'Min Annual: {min_overall} kWh\nMax Annual: {max_overall} kWh\nAvg Annual: {avg_overall} kWh') #Min to max spread of overall consumption

kwh_to_gbp = 0.335 #price of electricity in GBP
annual_standing_charge = 365 * 0.46 #daily standing charge, can't change this
average_bill = round(avg_overall * kwh_to_gbp + annual_standing_charge,2)
print(f'Average Bill: GBP {average_bill}') #average bill for each person

## Adding in challenges and nudges
print('\nGenerating new totals with nudges ...\n')
monthly_df = monthly_df[['User_ID',
                       'Profile',
                       'Washing Machine Consumption (kWh)',
                       'Dishwasher Consumption (kWh)',
                       'Dryer Consumption (kWh)',
                       'Fridge Consumption (kWh)',
                       'Annual Consumption (kWh)',
                       'Month1',
                       'Washing Machine Change?',
                       'New Washing Machine Consumption (kWh)',
                       'Dryer Change?',
                       'New Dryer Consumption (kWh)',
                       'Fridge Change?',
                       'New Fridge Consumption (kWh)']]

participation_rates = {
    'Tech-savvy': [0.5, 0.9, 0.8],
    'Traditional': [0.8, 0.4, 0.6],
    'Energy Champion': [1.0, 0.5, 0.7],
    'Laid-back': [0.1, 0.1, 0.2] #format: challenge, report, method
}

## INITIALISE MONTH1 ALL DATA

# generate whether user follows report or not
wash_col = [random.choices([1, 0], weights=[participation_rates[profile][1], 1 - participation_rates[profile][1]])[0] for profile in monthly_df['Profile']]
wash_col = wash_col[:len(monthly_df)] 
monthly_df.insert(len(monthly_df.columns) - 1, f'Month 1 Wash', wash_col) #insert the 1s and 0s if user follows report
dryer_col = [random.choices([1, 0], weights=[participation_rates[profile][1], 1 - participation_rates[profile][1]])[0] for profile in monthly_df['Profile']]
dryer_col = dryer_col[:len(monthly_df)] 
monthly_df.insert(len(monthly_df.columns) - 1, f'Month 1 Dry', dryer_col) #insert the 1s and 0s if user follows report
fridge_col = [random.choices([1, 0], weights=[participation_rates[profile][1], 1 - participation_rates[profile][1]])[0] for profile in monthly_df['Profile']]
fridge_col = fridge_col[:len(monthly_df)] 
monthly_df.insert(len(monthly_df.columns) - 1, f'Month 1 Fridge', fridge_col) #insert the 1s and 0s if user follows report
# Replace the 3 devices? Assume 1 in believing the report = replace all 3. Dishwasher no count.
monthly_df.loc[:, 'Washing']  = monthly_df.apply(lambda row:row['New Washing Machine Consumption (kWh)'] if row['Washing Machine Change?'] == 'Yes' and row[f'Month 1 Wash'] == 1 else row[f'Washing Machine Consumption (kWh)'], axis=1).astype(float)
monthly_df.loc[:, 'Dryer'] = monthly_df.apply(lambda row:row['New Dryer Consumption (kWh)'] if row['Dryer Change?'] == 'Yes' and row[f'Month 1 Dry'] == 1 else row[f'Dryer Consumption (kWh)'], axis=1).astype(float)
monthly_df.loc[:, 'Fridge'] = monthly_df.apply(lambda row:row['New Fridge Consumption (kWh)'] if row['Fridge Change?'] == 'Yes' and row[f'Month 1 Fridge'] == 1 else row[f'Fridge Consumption (kWh)'], axis=1).astype(float)

# Calculate new totals
monthly_df['Device Month1'] = monthly_df['Washing'] + monthly_df['Dryer'] + monthly_df['Fridge'] + monthly_df['Dishwasher Consumption (kWh)']
monthly_df['Always On Proportion Month1'] = monthly_df['Month1'] / (monthly_df['Device Month1'] + monthly_df['Month1'])
monthly_df['Always On Month1'] = monthly_df['Month1']

monthly_df = monthly_df[['User_ID',
                       'Profile',
                       'Annual Consumption (kWh)',
                       'Device Month1',
                       'Always On Month1',
                       'Always On Proportion Month1']]

challenge_mean = [0.015, 0, 0.014, 0, 0.015, 0, 0.036, 0, 0.018,0,0.156,0] #mean challenge reduction
challenge_sd = [0.005,0, 0.005,0, 0.005,0, 0.01,0,0.003,0,0.05,0] #SD for challenge
always_on = [0.2, 0.15, 0.1, 0.08, 0.05, 0.06, 0.02, 0.01, 0.04, 0.03, 0.04, 0.02] #max reduction per month

## BEGIN ITERATING THROUGH 12 MONTHS
for month in range(12):
    # CONSTRUCT 1 OR 0 INDICATING IF THE USER HAS FOLLOWED CHALLENGE/METHOD
    challenge_col = [random.choices([1, 0], weights=[participation_rates[profile][0], 1 - participation_rates[profile][0]])[0] for profile in monthly_df['Profile']]
    challenge_col = challenge_col[:len(monthly_df)] 
    method_col = [random.choices([1, 0], weights=[participation_rates[profile][2], 1 - participation_rates[profile][2]])[0] for profile in monthly_df['Profile']]
    method_col = method_col[:len(monthly_df)] 
    monthly_df.insert(len(monthly_df.columns) - 1, f'Month {month+1} Challenge', challenge_col)
    monthly_df.insert(len(monthly_df.columns) - 1, f'Month {month+1} Method', method_col)
    
    # CONSTRUCT REDUCTION COLUMNS
    monthly_df[f'Month {month+1} Challenge Reduction'] = np.random.normal(challenge_mean[month], challenge_sd[month], size=len(monthly_df))
    always_on_reduction = always_on[month]
    
    # APPLY REDUCTION IF USER HAS FOLLOWED
    monthly_df.loc[:, f'Device Month{month+2}'] = monthly_df.apply(lambda row: (1 - row[f'Month {month+1} Challenge Reduction']) * row[f'Device Month{month+1}'] if row[f'Month {month+1} Challenge'] == 1 else row[f'Device Month{month+1}'], axis=1)
    monthly_df.loc[:, f'Always On Month{month+2}'] = monthly_df.apply(lambda row: (1 - always_on_reduction) * row[f'Always On Month{month+1}'] if ((row[f'Month {month+1} Method'] == 1) and (row[f'Always On Proportion Month{month+1}'] > 0.2)) else row[f'Always On Month{month+1}'], axis=1)
    monthly_df.loc[:, f'Always On Proportion Month{month+2}'] = monthly_df[f'Always On Month{month+2}'] / (monthly_df[f'Device Month{month+2}'] + monthly_df[f'Always On Month{month+2}'])

    print(f'Always On Month {month+2}: ' , monthly_df[f'Always On Month{month+2}'].mean())
    print(f'Device Month {month+2}: ' , monthly_df[f'Device Month{month+2}'].mean())
monthly_df.to_csv('test.csv')

# print(monthly_df['Month1'].mean())
# print(monthly_df['Washing Machine Consumption (kWh)'].mean())
# print(monthly_df['Dishwasher Consumption (kWh)'].mean())
# print(monthly_df['Dryer Consumption (kWh)'].mean())
# print(monthly_df['Fridge Consumption (kWh)'].mean())

monthly_df['Overall']= monthly_df['Device Month13'] + monthly_df['Always On Month13']
max_overall = round(12*monthly_df['Overall'].max(),1)
min_overall = round(12*monthly_df['Overall'].min(),1) 
avg_overall = round(12*monthly_df['Overall'].mean(),1) 
print(f'Min Annual: {min_overall} kWh\nMax Annual: {max_overall} kWh\nAvg Annual: {avg_overall} kWh') #Min to max spread of overall consumption

kwh_to_gbp = 0.335 #price of electricity in GBP
annual_standing_charge = 365 * 0.46 #daily standing charge, can't change this
average_bill = round(avg_overall * kwh_to_gbp + annual_standing_charge,2)
print(f'Average Bill: GBP {average_bill}') #average bill for each person


# PLOTS FOR BEFORE AND AFTER

monthly_df['Overall'] = (12 * 0.335) * monthly_df['Overall'] + annual_standing_charge  # NEW CONSUMPTION WITH NUDGES
monthly_df['Annual Consumption (kWh)'] = 0.335 * monthly_df['Annual Consumption (kWh)'] + annual_standing_charge  # OLD CONSUMPTION WITHOUT NUDGES

fig, ax = plt.subplots(figsize=(15, 5))
boxplot_data = [monthly_df['Annual Consumption (kWh)'], monthly_df['Overall']]
boxplot_labels = ['Default (\u00A3)', 'With Nudges (\u00A3)']
bp = ax.boxplot(boxplot_data, vert=False, labels=boxplot_labels, showfliers=False)

# Calculate min, max, and median values
median_val = [monthly_df['Annual Consumption (kWh)'].median(), monthly_df['Overall'].median()]

# Add labels for min, max, and median
for i in range(len(boxplot_labels)):
    ax.text(median_val[i], i + 1, str(round(median_val[i], 2)), horizontalalignment='center', verticalalignment='center')

# Label whiskers
whiskers = [item for item in bp['whiskers']]
for w in whiskers:
    ax.text(w.get_xdata()[0], w.get_ydata()[0], str(round(w.get_xdata()[0], 2)),
            horizontalalignment='left', verticalalignment='top')
    ax.text(w.get_xdata()[1], w.get_ydata()[1], str(round(w.get_xdata()[1], 2)),
            horizontalalignment='right', verticalalignment='top')

ax.set_xlabel('Electricity Bill (\u00A3/annum)')
ax.set_title('Electricity Bill with and without Nudges (\u00A3/annum)')

plt.show()
