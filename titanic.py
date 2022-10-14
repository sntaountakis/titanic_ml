import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from data_processor import Splitter, AgeImputer


titanic_data = pd.read_csv('data/train.csv')

pc = Splitter(n_splits=1, test_size=0.2)
train_set, test_set = pc.split(titanic_data, ["Survived", "Pclass", "Sex"])


print(titanic_data)

# Split train, test data to [0.8, 0.2] ratio with similar distributions
# of Survived rates, Pclasses and Sexs
#split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
#for train_indices, test_indices in split.split(titanic_data, titanic_data[["Survived", "Pclass", "Sex"]]):
#    strat_train_set = titanic_data.loc[train_indices]
#    strat_test_set = titanic_data.loc[test_indices]


