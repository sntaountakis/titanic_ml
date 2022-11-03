import numpy as np
from collections import Counter

class NearestNeighbor():
    def __init__(self) -> None:
        pass

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y
    
    def predict(self, X, distance, k=1):
        n_tests = X.shape[0]

        Ypred = np.zeros(n_tests, dtype=self.ytr.dtype)
        for i in range(n_tests):
            if distance == 'L1':
                distances = np.sum(np.abs(self.Xtr - X[i,:]), axis=1)

            if distance == 'L2':
                distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis=1))
            
            min_index = np.argpartition(distances, k)
            Ypred[i] = Counter(self.ytr[min_index]).most_common(1)[0][0]
        
        return Ypred