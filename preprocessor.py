import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pipeline import Splitter, Imputer, Encoder, Dropper

class Preprocessor():
    def __init__(self, data, args) -> None:
        self.splits=args.splits
        self.test_size=args.test_size
        self.pc = Splitter(n_splits=self.splits, test_size=self.test_size)
        self.data = data
        self.important_features = args.important_features

    def preprocess(self):
        # Imputer: fills empty ages with age mean, Encoder: One-hot on gender and pclass
        # Dropper: Drops unecessary columns
        pipeline = Pipeline([('imputer', Imputer()),
                            ('encoder', Encoder()),
                            ('dropper', Dropper())])

        # Passes trainset through preprocessing pipeline
        self.data = pipeline.fit_transform(self.data)
        
        # Split train and test sets
        train_set, test_set = self.pc.split(self.data, self.important_features)

        # Split labels from data set
        X = train_set.drop(['Survived'], axis=1)
        Y = train_set['Survived']

        # Scale inputs on every feature s = (x - mean) / std
        scaler = StandardScaler()
        X_data = scaler.fit_transform(X)
        Y_data = Y.to_numpy()

        return X_data, Y_data
