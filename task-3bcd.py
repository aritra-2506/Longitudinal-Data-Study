import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
columns_to_skip = ['ID']
dataset = pd.read_csv('data_disha.csv', usecols=lambda x: x not in columns_to_skip, skiprows=3)

bins = [7, 16, 24, 28, 32, 37, 100]
label = [1, 2, 3, 4, 5, 6]
dataset['binned'] = pd.cut(dataset['GA'], bins, labels=label, include_lowest=True)

dt_x= dataset[['binned']]
dt_ga=dataset[['GA']]
r = datetime.datetime.strptime(dt_ga + '-1', "%Y-W%W-%w")

