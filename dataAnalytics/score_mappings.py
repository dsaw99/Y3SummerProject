import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from collections import Counter
####### PART I: MISCONCEPTIONS ABOUT METHODS #######

# Q1: Conscious effort to reduce usage
action_score = {
    'Definitely not': 0,
    'Probably not': 0.25,
    'Might or might not': 0.5,
    'Probably yes': 0.75,
    'Definitely yes': 1
}

# Q2: 3 methods to reduce from a list of 15
method_savings = {
    'Turning off lights and appliances': 0,
    'Unplugging electronics': 1,
    'Turning off plugs': 1,
    'Using a programmable thermostat': 4,
    'Using thermal curtains, blinds, or shades for better insulation': 0,
    'Switching to energy efficient lighting such as LEDs or CFLs': 4,
    'Using natural light': 2,
    'Running full loads in dishwasher/washing machine': 3,
    'Using energy-saving modes i.e. eco-mode': 3,
    'Washing clothes in cold water': 3,
    'Air drying clothes': 4,
    'Spending less time in the shower': 2,
    'Swapping baths for showers': 2,
    'Insulating your hot water tank': 3,
    'Charging your EV during off-peak hours': 0
}

# Q3: Opinion on effectiveness
effectiveness_score = {
    'Not effective at all': 0,
    'Slightly effective': 1,
    'Moderately effective': 2,
    'Very effective': 3,
    'Extremely effective': 4
}

effectiveness_table = [
    [1, 0.75, 0.5, 0.25, 0],
    [0.75, 1, 0.75, 0.5, 0.25],
    [0.5, 0.75, 1, 0.75, 0.5],
    [0.25, 0.5, 0.75, 1, 0.75],
    [0, 0.25, 0.5, 0.75, 1]
]

def energy_saving_knowledge(df):
    # Create a new column 'Score' to store the calculated scores
    df['Score'] = np.nan

    # Create a dictionary to store the frequency of each choice
    choice_frequency = {}

    # Create a dictionary to the tier frequencies 
    tier_frequency = {
        'Tier 0': 0,
        'Tier 1': 0,
        'Tier 2': 0,
        'Tier 3': 0,
        'Tier 4': 0
    }

    # Iterate over each row and calculate the score
    for index, row in df.iterrows():
        choices = str(row['Q2']).split(',') if pd.notnull(row['Q2']) else []
        # The thermal curtains choice contains commas in them so this is a hardcoded fix for it
        if len(choices) > 3:
            new_choices = []
            for choice in choices:
                if choice.strip() not in ['Using thermal curtains', 'blinds', 'or shades for better insulation']:
                    new_choices.append(choice.strip())

            new_choices.append('Using thermal curtains, blinds, or shades for better insulation')
            choices = new_choices

        # Update the frequency of each choice
        for choice in choices:
            choice_frequency[choice] = choice_frequency.get(choice, 0) + 1
            if(choice != 'Other'):
                tier_frequency[f'Tier {method_savings[choice]}'] += 1

        effectiveness = row['Q3 ']
        score = sum(effectiveness_table[effectiveness_score[effectiveness]][method_savings.get(choice, 0)] for choice in choices)
        df.at[index, 'Score'] = score
    
    print(tier_frequency) #Print each tier
    # Plot the histogram of scores
    plt.hist(df['Score'], bins=10, edgecolor='black')

    # Set the labels and title
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Spread of Scores')

    # Display the plot
    plt.show()

    # Plot the frequency of each choice
    choices = list(choice_frequency.keys())
    frequencies = list(choice_frequency.values())
    plt.bar(choices, frequencies)
    plt.xlabel('Choice')
    plt.ylabel('Frequency')
    plt.title('Frequency of Choices')
    plt.xticks(rotation=90)
    plt.show()

    print("Overall mean score:", np.mean(df['Score']))
    energy_df = pd.DataFrame({'Method': list(choice_frequency.keys()), 'Frequency': list(choice_frequency.values())})
    energy_df = energy_df.sort_values('Frequency', ascending=False)
    energy_df.to_csv('output/method_frequency.csv', index=False)
    return np.array(df['Score'])

device_tiers = {
    'S-tier': ['Washing machine', 'Oven', 'Refrigerator'],
    'A-tier': ['Dryer', 'Dishwasher', 'Water heater'],
    'B-tier': ['Large televisions','Desktop computers','Lighting','Game consoles'],
    'C-tier': ['Kettle','Microwave'],
    'D-tier': ['Chargers', 'Printers', 'Hair dryer', 'Coffee machines', 'Toaster']
}

# Q4: Identification of 3 devices from 17
def device_knowledge(df):
    # Calculate the scores for each respondent
    scores_yes = []
    scores_no = []
    scores = []
    device_frequency = {}

    tier_frequency = {
        'S-tier': 0,
        'A-tier': 0,
        'B-tier': 0,
        'C-tier': 0,
        'D-tier': 0
    }
    for index, row in df.iterrows():
        # Check if the user can identify the devices
        if row['Q4'] == 'Yes':
            score = 3
            # Get the selected devices from Q5
            selected_devices = str(row['Q5']).split(',')
        else:
            score = 0
            # Get the selected devices from Q5.1
            selected_devices = str(row['Q5.1']).split(',')

        # Count the number of correct guesses in each tier
        for tier, devices in device_tiers.items():
            for device in selected_devices:
                if device.strip() in devices:
                    if tier == 'S-tier':
                        score += 3
                        tier_frequency['S-tier'] += 1
                    elif tier == 'A-tier':
                        score += 2
                        tier_frequency['A-tier'] += 1
                    elif tier == 'B-tier':
                        score += 0
                        tier_frequency['B-tier'] += 1
                    elif tier == 'C-tier' and row['Q4'] == 'Yes':
                        score -= 1
                        tier_frequency['C-tier'] += 1
                    elif tier == 'D-tier' and row['Q4'] == 'Yes':
                        score -= 2
                        tier_frequency['D-tier'] += 1
                    elif tier == 'C-tier' and row['Q4'] == 'No':
                        tier_frequency['C-tier'] += 1
                    elif tier == 'D-tier' and row['Q4'] == 'No':
                        tier_frequency['D-tier'] += 1

        if row['Q4'] == 'Yes':
            scores_yes.append(score)
        else:
            scores_no.append(score)
        scores.append(score)

        # Update the frequency of each selected device
        for device in selected_devices:
            device_frequency[device.strip()] = device_frequency.get(device.strip(), 0) + 1
        
    print(tier_frequency)
    # Plot the histogram of scores for 'Yes' and 'No' respondents separately
    plt.hist(scores_yes, bins=10, edgecolor='black', alpha=0.5, label='Q4: Yes')
    plt.hist(scores_no, bins=10, edgecolor='black', alpha=0.5, label='Q4: No')

    # Set the labels and title
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Spread of Scores')

    # Add a legend
    plt.legend()

    # Display the plot
    plt.show()

    # Plot the frequency of each selected device
    devices = list(device_frequency.keys())
    frequencies = list(device_frequency.values())
    plt.bar(devices, frequencies)
    plt.xlabel('Device')
    plt.ylabel('Frequency')
    plt.title('Frequency of Selected Devices')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    print("Mean score for 'Yes' respondents:", np.mean(scores_yes))
    print("Mean score for 'No' respondents:", np.mean(scores_no))
    print("Overall mean score:", np.mean(scores))

    device_df = pd.DataFrame({'Device': list(device_frequency.keys()), 'Frequency': list(device_frequency.values())})
    device_df = device_df.sort_values('Frequency', ascending=False)
    device_df.to_csv('output/device_frequency.csv', index=False)
    return scores

def overall_knowledge(energy_saving_score, device_score):
    weight_energy = 5 #rated 0 to 3 --> 0 to 15
    offset_energy = 0
    weight_device = 1 #rated -3 to 12, with small amounts of negative
    offset_device = 3 #0 to 15
    ratio = 1/30 #normalise to 0-1
    
    overall_scores = []
    for e_score, d_score in zip(energy_saving_score, device_score):
        overall_scores.append(ratio * (weight_energy * e_score + offset_energy + weight_device * d_score + offset_device))
    plt.hist(overall_scores, bins=10, edgecolor='black', alpha=0.5, label='Overall')

    # Set the labels and title
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Spread of Scores')

    # Add a legend
    plt.legend()

    # Display the plot
    plt.show()
    print("The overall score is:", np.mean(overall_scores))
    return overall_scores

def distance_from_max(array, max_score):
    dist_array = []
    total_dist = 0
    for value in array:
        dist_array.append(max_score - value)
        total_dist += max_score - value
    print("Mean distance from 1:", total_dist / len(dist_array))
    print(dist_array)
    return dist_array

def export_csv(df, filename):
    df.to_csv(filename, index=False)

# Q6 onwards: General count
def count_frequency(df):
    selected_columns = ['Q1', 'Q2', 'Q2b', 'Q3 ', 'Q4', 'Q5', 'Q5.1', 'Q5b', 'Q6', 'Q7', 'Q8 ', 'Q9', 'Q19', 'Q19.1', 'Q20', 'Q21', 'Q22', 'Q23', 'Q19.2', 'Q20.1', 'Q10', 'Q11']
    selected_data = df[selected_columns].copy()

    # Create a writer object to save multiple sheets in a single file
    writer = pd.ExcelWriter('survey/survey_frequency.xlsx', engine='xlsxwriter')

    # Q2 commas problem
    selected_data['Q2'] = selected_data['Q2'].replace({'Using thermal curtains, blinds, or shades for better insulation':
                                                       'Using thermal curtains; blinds or shades for better insulation'}, regex=True)
    # Define column descriptions
    column_descriptions = {
        'Q1': 'Do you make a conscious effort to save energy around the house?',
        'Q2': 'Could you provide three examples of how you do this?',
        'Q3 ': 'Do you think these actions are effective in reducing your energy costs?', 
        'Q4': 'Can you identify the three devices in your home that use the most energy?', 
        'Q5': 'If yes, could you give us your top three?',
        'Q5.1': 'If your answer is no, could you make an educated guess and give us your top three?',
        'Q5b': 'If other, please specify:',
        'Q6': 'How often is the oven used in your household?',
        'Q7': 'When is the oven mostly used?',
        'Q8 ': 'How often is your washing machine used?',
        'Q9': 'When is the washing machine mostly used?',
        'Q19': 'Have you heard about Smart Meters before?',
        'Q19.1': 'Do you have a Smart Meter installed in your house?',
        'Q20': 'If yes, how often do you check your Smart Meter?',
        'Q21': 'Do you have concerns regarding Smart Meters, such as them being invasive?',
        'Q22': 'Are the costs involved with a Smart Meter a concern?',
        'Q23': 'Have you heard about energy monitors?',
        'Q19.2': 'Would you consider installing an energy monitor?',
        'Q20.1': 'If yes, would you listen to the advice generated in line with the data it collects?',
        'Q10': 'Which of the following age groups do you fall into?',
        'Q11': 'How many residents are there in your household? (including yourself)',
    }

    # Calculate the frequency of survey responses for each column and save each as a separate sheet
    for column in selected_columns:
        if column in column_descriptions:
            column_description = column_descriptions[column]
        else:
            column_description = 'No description available'

        if column == 'Q5b':
            # Calculate frequency for Q5b
            column_values = selected_data['Q5b'].str.split(',\s*').explode()
            column_frequency = column_values.value_counts().reset_index()
            column_frequency.columns = ['Response', 'Frequency']
            column_frequency.to_excel(writer, sheet_name=column, index=False, header=['Response', 'Frequency'])

            # Calculate combined frequency for Q5, Q5.1, and Q5b
            combined_values = pd.concat([selected_data['Q5'], selected_data['Q5.1'], selected_data['Q5b']], ignore_index=True).dropna()
            combined_frequency = combined_values.str.split(',\s*').explode().value_counts().reset_index()
            combined_frequency.columns = ['Response', 'Frequency']
            combined_frequency.to_excel(writer, sheet_name='Q5comb', index=False, header=['Response', 'Frequency'])

        else:
            column_values = selected_data[column].str.split(',\s*').explode()
            column_frequency = column_values.value_counts().reset_index()
            column_frequency.columns = ['Response', 'Frequency']
            column_frequency.to_excel(writer, sheet_name=column, index=False, header=[column_description, 'Frequency'])

    # Save the Excel file
    writer.save()

    print("Frequency data saved to 'survey/survey_frequency.xlsx'")

