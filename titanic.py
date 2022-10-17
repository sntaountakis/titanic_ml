import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyparsing import traceParseAction
import seaborn as sb
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pipeline import Splitter, Imputer, Encoder, Dropper
from preprocessor import Preprocessor
import argparse

def train(args):
    titanic_data = pd.read_csv('data/train.csv')
    pp = Preprocessor(data=titanic_data, args=args)

    X_data, Y_data = pp.preprocess()
    print(X_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", help="Number of train/test splits", type=int, default=1)
    parser.add_argument("--test_size", help="Percentage of data to be split in test data", type=float, default=0.2)
    parser.add_argument("--important_features", nargs='+', default=["Survived", "Pclass", "Sex"])
    args = parser.parse_args()
    print(args.splits)
    train(args)