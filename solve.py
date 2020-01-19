from math import log
from copy import deepcopy
import numpy as np
from sklearn.preprocessing import normalize
import pandas as pd

class AdaBoost:

    def __init__(self, classifier, n):
        def runClear():
            self.classifiers =  [deepcopy(classifier) for i in range(n)]
            self.classifierWights = np.array([])
            self.classifierErrors = np.array([])
        self.clear = runClear
        self.clear()

    def fit(self, X, y):
        self.clear()

        l = len(X)
        w = np.repeat(1 / l, l)

        for c in self.classifiers:
            c.fit(X, y, sample_weight=w)
            yPredicted = c.predict(X)
            N = sum([w[i] for i in range(l) if y[i] != yPredicted[i]])
            np.append(self.classifierErrors, N)

            eps = 1e-6
            if N < eps or N > 1 - eps:
                self.classifiers = np.array([c])
                self.w = [1 if N < eps else -1]
                break

            b = 0.5 * log((1 - N) / N)
            np.append(self.classifierWights, b)

            w = np.multiply(w, np.exp(-b * np.multiply(y, yPredicted)))
            w = normalize(a.reshape(-1,1)).reshape(1,-1)[0]
            
    def predict(self, X):
        preds = np.zeros(len(X))
        for w, c in zip(self.w, self.classifiers):
            preds += w * c.predict(X)
        return np.sign(preds)

def readDataset(filename):
    x, labels = load_dataset(filename)
    y = process_labels(labels)
    return x, y

def load_dataset(filename):
    df = pd.read_csv(filename).sample(frac=1)
    return df[['x', 'y']].to_numpy(), df['class'].to_numpy()


def process_labels(labels):
    return np.array(list(map(lambda label: 1 if label == 'P' else -1, labels)))

x, y = readDataset("Bayes/chips.csv")