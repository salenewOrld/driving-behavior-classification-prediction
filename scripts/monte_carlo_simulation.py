import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score

class MonteCarloSimulation:
    def __init__(self, df, target_col, n_splits=10, test_size=0.2, random_state=9999):
        self.df = df
        self.target_col = target_col
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
        self.accuracies = []

    def run_simulation(self):
        sss = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=self.test_size, random_state=self.random_state)
        for train_index, test_index in sss.split(self.df, self.df[self.target_col]):
            X_train, X_test = self.df.iloc[train_index], self.df.iloc[test_index]
            y_train, y_test = X_train[self.target_col], X_test[self.target_col]

            clf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            clf.fit(X_train.drop(self.target_col, axis=1), y_train)

            y_pred = clf.predict(X_test.drop(self.target_col, axis=1))
            self.accuracies.append(recall_score(y_test, y_pred, average='macro'))

    def get_results(self):
        return np.array(self.accuracies)

if __name__ == '__main__':
    df = pd.read_csv('/Users/salene/Documents/GitHub/driving-behavior-classification-prediction/datasets/driving-behavior/train_motion_data.csv')
    target_col = 'Class'

    mcs = MonteCarloSimulation(df, target_col)
    mcs.run_simulation()
    results = mcs.get_results()
    print('Recalls:', results)
    print('Mean accuracy:', np.mean(results))
    print('Standard deviation:', np.std(results))
