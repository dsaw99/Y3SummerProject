import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from io import BytesIO
from io import StringIO


class Fridge:
  def __init__(self):
    self.fridge_dic = {
      'under counter': {
        'D': 87,
        'E': 91, #92 136L # 90 133L 179               # 0.669, 0.6842  -> 0.6766
        'F': 113  #114 136L # 113 131L # 142 105L     # 0.838 0.863    -> 0.8505
      },
      'freestanding' : { # 2: 240L, 3:260, 4: 300, 5:370
          #'C': 157, #104 260L 1629p
        'D': 184, # 130 244L 1300p #129 260L 849p     #0.496 0.533   -> 0.5145
        'E': 219, # 212 #143 365L 579p                #0.58
        'F': 263 #287 255L 500p                       #1.125
      },
      'american' : { # have 'super freeze' function'
        #'C': 225, # 225 409L 1699p #163 302L 729p
        'D': 269, #223  311L 899p                                   #0.729
        'E': 337,  #287 291L 529p  #298 306L 699p #323 371L 729     #0.871
        'F': 406    #392 327L #395 409L 1099p                       #1.199
      }
    }
    self.fridge_cost = {
      'under counter': np.random.normal(200, 50),
      'freestanding': np.random.normal(600, 100)
      #'american': np.random.normal(1.85, 0.02)
    }
    self.type_coeff = {
    'under counter': 0.6766,
    'freestanding': 0.5145
    }

    self.capacity_dic = {
      1: np.random.normal(113,5),
      2: np.random.normal(170,5),
      3: np.random.normal(226,5),
      4: np.random.normal(283,5),
      5: np.random.normal(339,5)
      #'american': np.random.normal(1.85, 0.02)
    }

    self.avg_fridge_lifespan = 11
    self.common_ratio = 1.01
    self.fridge_lifespan = self.find_lifespan(self.avg_fridge_lifespan, self.common_ratio)

  def find_lifespan(self, lifespan, avg_fridge_lifespan = 11):

    lifespan_coeff = 0
    for i in range(lifespan):
        term = avg_fridge_lifespan ** (i + 1)
        lifespan_coeff += term

    return lifespan_coeff

  def future_lifespan(self, year_used, avg_fridge_lifespan = 11):
    end_year = year_used + avg_fridge_lifespan
    lifespan_coeff2 = 0

    for i in range(year_used, end_year):
        term2 = self.common_ratio ** (i + 1)
        lifespan_coeff2 += term2

    return lifespan_coeff2

  def payback_period(self, n, users, device):
    curr_cost = self.fridge_dic[users.loc[n,'Fridge Type']][users.loc[n,'Fridge Rating']] # annual
    future_cost = device.loc[n,'Annual Running Cost']
    threshold = device.loc[n,'Substitute Cost'] / (curr_cost - future_cost)
    return threshold

  def process_fridge(self, users, users_size, fridge_report):

    for n in range(users_size):

      coeff = self.future_lifespan(users.loc[n,'Fridge Years Used'], 11) # from now to future 11 years
      fridge_report.loc[n,'Lifetime Cost'] = np.random.normal(self.fridge_dic[users.loc[n,'Fridge Type']][users.loc[n,'Fridge Rating']], 5) * coeff

      # suggest different types for different households
      if users.loc[n,'Household_Size'] == 1:
        # sugget user use E rating under counter fridge
        if users.loc[n,'Fridge Rating'] != 'E':
          fridge_report.loc[n, 'recommended type'] = 'under counter'
          fridge_report.loc[n, 'suggested capacity'] = self.capacity_dic[1]
          fridge_report.loc[n,'Substitute Cost'] = self.fridge_cost[fridge_report.loc[n,'recommended type']]
          fridge_report.loc[n,'Annual Running Cost'] = fridge_report.loc[n, 'suggested capacity'] * self.type_coeff['under counter']

          threshold = self.payback_period(n, users, fridge_report)
          if (25 - users.loc[n,'Fridge Years Used']) > threshold:
            fridge_report.loc[n, 'decision'] = True # suggest replacing devices
            # new cost = price of new one + appr. running cost for 11 years (avg life span, coeff)
            fridge_report.loc[n,'New Cost'] = fridge_report.loc[n,'Substitute Cost'] + fridge_report.loc[n,'Annual Running Cost'] * self.fridge_lifespan

        else:
          fridge_report.loc[n, 'decision'] = False
          fridge_report.loc[n, 'recommended type'] = np.nan
          fridge_report.loc[n, 'suggested capacity'] = np.nan
          fridge_report.loc[n, 'New Cost'] = np.nan

      else:
        household = users.loc[n,'Household_Size']
        if users.loc[n,'Fridge Rating'] != 'D':
          fridge_report.loc[n, 'recommended type'] = 'freestanding'
          fridge_report.loc[n, 'suggested capacity'] = self.capacity_dic[household]
          fridge_report.loc[n,'Substitute Cost'] = self.fridge_cost[fridge_report.loc[n,'recommended type']]
          fridge_report.loc[n,'Annual Running Cost'] = fridge_report.loc[n, 'suggested capacity'] * self.type_coeff['freestanding']

          threshold = self.payback_period(n, users, fridge_report)
          if threshold > 0 and (threshold < 11 or (20-users.loc[n,'Fridge Years Used']) > threshold):
            fridge_report.loc[n, 'decision'] = True
            fridge_report.loc[n,'New Cost'] = fridge_report.loc[n,'Substitute Cost'] + fridge_report.loc[n,'Annual Running Cost'] * self.fridge_lifespan

        else:
          fridge_report.loc[n, 'decision'] = False
          fridge_report.loc[n, 'recommended type'] = np.nan
          fridge_report.loc[n, 'suggested capacity'] = np.nan
          fridge_report.loc[n, 'New Cost'] = np.nan

    return fridge_report

  def fridge_report(self, users, users_size, fridge_report, suggestion_list):

    fridge_report = users[['Household_Size']].copy()
    fridge_report = self.process_fridge(users, users_size, fridge_report)
    fridge_report = fridge_report[['Household_Size', 'Lifetime Cost', 'recommended type', 'suggested capacity', 'New Cost', 'decision']]

    list_index = 0

    for n in range(users_size):

      if fridge_report.loc[n,'New Cost'] < fridge_report.loc[n,'Lifetime Cost'] and fridge_report.loc[n, 'decision']:
        suggestion_list.loc[list_index,'Household_Size'] = fridge_report.loc[n, 'Household_Size']
        suggestion_list.loc[list_index,'recommended type'] = fridge_report.loc[n, 'recommended type']
        suggestion_list.loc[list_index,'suggested capacity (L)'] = round(fridge_report.loc[n, 'suggested capacity'])
        suggestion_list.loc[list_index,'current plan cost'] = round(fridge_report.loc[n, 'Lifetime Cost'])
        suggestion_list.loc[list_index,'new plan cost'] = round(fridge_report.loc[n, 'New Cost'])
        suggestion_list.loc[list_index,'Appr. Saving'] = round(fridge_report.loc[n, 'Lifetime Cost'] - fridge_report.loc[n, 'New Cost'])
        list_index += 1

    print(fridge_report)
    print(suggestion_list)
    annual_total_saving = suggestion_list['Appr. Saving'].sum() / 11


    return suggestion_list, list_index, annual_total_saving