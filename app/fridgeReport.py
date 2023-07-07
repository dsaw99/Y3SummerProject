import pandas as pd
import numpy as np

class Fridge:
  def __init__(self):
    self.unit_price = 0.335
    self.fridge_dic = {
      'under counter': {
        'D': 87,
        'E': 91,
        'F': 113 
      },
      'freestanding' : {
        'C': 157,
        'D': 184,
        'E': 219,
        'F': 263
      },
      'american' : {
        'D': 269,
        'E': 337,
        'F': 406
      }
    }
    self.fridge_cost = {
      'under counter': np.random.normal(200, 50),
      'freestanding': np.random.normal(600, 100)
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

    for i in range(int(year_used), int(end_year)):
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

      coeff = self.future_lifespan(users.loc[n,'Fridge Years Used'], self.common_ratio) # from now to future 11 years
      fridge_report.loc[n,'Lifetime Cost'] = np.random.normal(self.fridge_dic[users.loc[n,'Fridge Type']][users.loc[n,'Fridge Rating']], 5) * coeff * self.unit_price * self.avg_fridge_lifespan

      if users.loc[n,'Household_Size'] == 1:
        if users.loc[n,'Fridge Rating'] == 'F' or users.loc[n, 'Fridge Type'] != 'under counter':
          fridge_report.loc[n, 'recommended type'] = 'under counter'
          fridge_report.loc[n, 'suggested capacity'] = self.capacity_dic[1]
          fridge_report.loc[n,'Substitute Cost'] = self.fridge_cost[fridge_report.loc[n,'recommended type']]
          fridge_report.loc[n,'Annual Running Cost'] = fridge_report.loc[n, 'suggested capacity'] * self.type_coeff['under counter'] * self.unit_price
          fridge_report.loc[n,'New Consumption (kWh)'] = fridge_report.loc[n,'Annual Running Cost'] / 12 / self.unit_price

          threshold = self.payback_period(n, users, fridge_report)

          if (25 - users.loc[n,'Fridge Years Used']) > threshold:
            fridge_report.loc[n, 'decision'] = True
            fridge_report.loc[n,'New Cost'] = fridge_report.loc[n,'Substitute Cost'] + fridge_report.loc[n,'Annual Running Cost'] * self.fridge_lifespan
          else:
            fridge_report.loc[n, 'decision'] = False
        else:
          fridge_report.loc[n, 'decision'] = False

      else:
        household = users.loc[n,'Household_Size']
        if users.loc[n,'Fridge Rating'] != 'D' and users.loc[n, 'Fridge Type'] != 'under counter':
          fridge_report.loc[n, 'recommended type'] = 'freestanding'
          fridge_report.loc[n, 'suggested capacity'] = self.capacity_dic[household]
          fridge_report.loc[n,'Substitute Cost'] = self.fridge_cost[fridge_report.loc[n,'recommended type']]
          fridge_report.loc[n,'Annual Running Cost'] = fridge_report.loc[n, 'suggested capacity'] * self.type_coeff['freestanding'] * self.unit_price
          fridge_report.loc[n,'New Consumption (kWh)'] = fridge_report.loc[n,'Annual Running Cost'] / 12 / self.unit_price

          threshold = self.payback_period(n, users, fridge_report)
        
          if threshold > 0 and (threshold < 11 or (20-users.loc[n,'Fridge Years Used']) > threshold):
            fridge_report.loc[n, 'decision'] = True
            fridge_report.loc[n,'New Cost'] = fridge_report.loc[n,'Substitute Cost'] + fridge_report.loc[n,'Annual Running Cost'] * self.fridge_lifespan
          else:
            fridge_report.loc[n, 'decision'] = False

        else:
          fridge_report.loc[n, 'decision'] = False

    return fridge_report

  def fridge_report(self, users, users_size, fridge_report, suggestion_list):

    fridge_report = users[['User_ID', 'Household_Size']].copy()
    fridge_report = self.process_fridge(users, users_size, fridge_report)
    print(fridge_report)
    if fridge_report.loc[0, 'decision'] == True:
      fridge_report = fridge_report[['User_ID', 'Household_Size', 'Lifetime Cost', 'recommended type', 'suggested capacity', 'New Cost', 'New Consumption (kWh)','decision']]

      list_index = 0

      for n in range(users_size):
        print(fridge_report)
        if fridge_report.loc[n,'New Cost'] < fridge_report.loc[n,'Lifetime Cost'] and fridge_report.loc[n, 'decision']:
          suggestion_list.loc[list_index,'User_ID'] = round(fridge_report.loc[n,'User_ID'], 0)
          suggestion_list.loc[list_index,'Household_Size'] = round(fridge_report.loc[n, 'Household_Size'], 0)
          suggestion_list.loc[list_index,'recommended type'] = fridge_report.loc[n, 'recommended type']
          suggestion_list.loc[list_index,'suggested capacity (L)'] = round(fridge_report.loc[n, 'suggested capacity'])
          suggestion_list.loc[list_index,'current plan cost'] = round(fridge_report.loc[n, 'Lifetime Cost'])
          suggestion_list.loc[list_index,'new plan cost'] = round(fridge_report.loc[n, 'New Cost'])
          suggestion_list.loc[list_index,'New Consumption (kWh)'] = (fridge_report.loc[n, 'New Consumption (kWh)'])
          suggestion_list.loc[list_index,'Appr. Saving'] = round(fridge_report.loc[n, 'Lifetime Cost'] - fridge_report.loc[n, 'New Cost'])
          list_index += 1

      annual_total_saving = suggestion_list['Appr. Saving'].sum() / self.avg_fridge_lifespan


      return suggestion_list, list_index, annual_total_saving
    
    else:
      return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()