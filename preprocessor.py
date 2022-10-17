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
        
        # Split train and test sets
        train_set, test_set = self.pc.split(self.data, self.important_features)

        # Imputer: fills empty ages with age mean, Encoder: One-hot on gender and pclass
        # Dropper: Drops unecessary columns
        pipeline = Pipeline([('imputer', Imputer()),
                            ('encoder', Encoder()),
                            ('dropper', Dropper())])


        # Passes trainset through preprocessing pipeline
        test_set = pipeline.fit_transform(test_set)
        train_set = pipeline.fit_transform(train_set)
        

        # Scale inputs on every feature s = (x - mean) / std
        scaler = StandardScaler()

        # Split labels from data set
        X_train = train_set.drop(['Survived'], axis=1)
        Y_train = train_set['Survived'].to_numpy()
        X_train = scaler.fit_transform(X_train)

        X_test = test_set.drop(['Survived'], axis=1)
        Y_test =  test_set['Survived'].to_numpy()
        X_test = scaler.fit_transform(X_test)
        
        return X_train, Y_train, X_test, Y_test
