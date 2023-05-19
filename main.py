import dataProcessing
import tarfile
import csv
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('/home/mp1820/Y3SummerProject/output/User 1 combined data.csv')
dfWeek, dfWeekend = dataProcessing.split_weekdays_weekends(df)
dataProcessing.plot_hourly_linegraph(dfWeek, 'line graph User 1 Weekdays')
dataProcessing.plot_hourly_linegraph(dfWeekend, 'line graph User 1 Weekend')
dataProcessing.plot_hourly_boxplot(dfWeek, 'Boxplot User 1 Weekdays')
dataProcessing.plot_hourly_boxplot(dfWeekend, 'Boxplot User 1 Weekend')

df = pd.read_csv('/home/mp1820/Y3SummerProject/output/User 2 combined data.csv')
dfWeek, dfWeekend = dataProcessing.split_weekdays_weekends(df)
dataProcessing.plot_hourly_linegraph(dfWeek, 'line graph User 2 Weekdays')
dataProcessing.plot_hourly_linegraph(dfWeekend, 'line graph User 2 Weekend')
dataProcessing.plot_hourly_boxplot(dfWeek, 'Boxplot User 2 Weekdays')
dataProcessing.plot_hourly_boxplot(dfWeekend, 'Boxplot User 2 Weekend')

df = pd.read_csv('/home/mp1820/Y3SummerProject/output/data27_overall.csv')
dfWeek, dfWeekend = dataProcessing.split_weekdays_weekends(df)
dataProcessing.plot_hourly_linegraph(dfWeek, 'line graph NY Weekdays')
dataProcessing.plot_hourly_linegraph(dfWeekend, 'line graph NY Weekend')
dataProcessing.plot_hourly_boxplot(dfWeek, 'Boxplot NY Weekdays')
dataProcessing.plot_hourly_boxplot(dfWeekend, 'Boxplot NY Weekend')