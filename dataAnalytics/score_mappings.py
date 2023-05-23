import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        effectiveness = row['Q3 ']
        score = sum(effectiveness_table[effectiveness_score[effectiveness]][method_savings.get(choice, 0)] for choice in choices)
        df.at[index, 'Score'] = score

    # Plot the histogram of scores
    plt.hist(df['Score'], bins=10, edgecolor='black')

    # Set the labels and title
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Spread of Scores')

    # Display the plot
    plt.show()

    print("Overall mean score:", np.mean(df['Score']))
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
                    elif tier == 'A-tier':
                        score += 2
                    elif tier == 'B-tier':
                        score += 0
                    elif tier == 'C-tier' and row['Q4'] == 'Yes':
                        score -= 1
                    elif tier == 'D-tier' and row['Q4'] == 'Yes':
                        score -= 2

        if row['Q4'] == 'Yes':
            scores_yes.append(score)
        else:
            scores_no.append(score)
        scores.append(score)
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

    print("Mean score for 'Yes' respondents:", np.mean(scores_yes))
    print("Mean score for 'No' respondents:", np.mean(scores_no))
    print("Overall mean score:", np.mean(scores))
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
    print(overall_scores)
    plt.hist(overall_scores, bins=10, edgecolor='black', alpha=0.5, label='Overall')

    # Set the labels and title
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Spread of Scores')

    # Add a legend
    plt.legend()

    # Display the plot
    plt.show()
    return overall_scores