import dataProcessing as dataProcessing
import tarfile
import csv
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

data = dataProcessing.Dataset("output/User3.csv")
print(data.getScore())