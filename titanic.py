import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessor import Preprocessor
from nearest_neighbor import NearestNeighbor
import argparse

def train(args):
    titanic_data = pd.read_csv('data/train.csv')
    pp = Preprocessor(data=titanic_data, args=args)

    X_train, Y_train, X_test, Y_test = pp.preprocess()
    nn = NearestNeighbor()
    nn.train(X_train, Y_train)
    Y_test_pred = nn.predict(X_test, args.distance_equation)

    print('Accuracy: {}'.format(np.mean(Y_test == Y_test_pred)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", help="Number of train/test splits", type=int, default=1)
    parser.add_argument("--test_size", help="Percentage of data to be split in test data", type=float, default=0.2)
    parser.add_argument("--important_features", nargs='+', default=["Survived", "Pclass", "Sex"])
    parser.add_argument("--distance_equation", default='L1', choices=['L1', 'L2'])
    args = parser.parse_args()
    print(args.splits)
    train(args)