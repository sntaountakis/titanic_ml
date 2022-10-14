from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

class Splitter():
    
    def __init__(self, n_splits, test_size) -> None:
        self.splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    
    def split(self, dataset: pd.DataFrame, features: List[str]):
        for train_indices, test_indices in self.splitter.split(dataset, dataset[features]):
            strat_train_set = dataset.loc[train_indices]
            strat_test_set = dataset.loc[test_indices]
        return strat_train_set, strat_test_set
    
class AgeImputer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        imputer = SimpleImputer(strategy="mean")
        X['Age'] = imputer.fit_transform(X['Age'])
        
        return X
