import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import random
class MonteCarloSimulation:
    def __init__(self, df, target_col, n_splits=10, test_size=0.2, random_state=9999):
        self.df = df
        self.target_col = target_col
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
        self.accuracies = []
        self.recalls = []
        self.f1_macros = []
        self.precisions = []

    def run_simulation(self):
        sss = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=self.test_size, random_state=self.random_state)
        num = 1
        self.df['Class'] = self.df['Class'].astype('category').cat.codes
        for train_index, test_index in sss.split(self.df, self.df[self.target_col]):
            print('-'*50)
            print(f'Running at sample number -> {num}'.center(50))
            print(f'Models to train XGB, RFClassifier'.center(50))
            scores_xgb = []
            scores_rf = []
            X_train, X_test = self.df.iloc[train_index], self.df.iloc[test_index]
            y_train, y_test = X_train[self.target_col], X_test[self.target_col]
            self.models = [XGBClassifier(), RandomForestClassifier()]
            for j in range(2): 
                self.models[j].fit(X_train.drop(self.target_col, axis=1), y_train)
                y_pred = self.models[j].predict(X_test.drop(self.target_col, axis=1))
                recall = recall_score(y_test, y_pred, average='macro')
                f1 = f1_score(y_test, y_pred, average='macro')
                precision = precision_score(y_test, y_pred, average='macro')
                acc = accuracy_score(y_test, y_pred)
                if j == 1:
                    scores_rf.append(recall)
                    scores_rf.append(acc)
                    scores_rf.append(precision)
                    scores_rf.append(f1)
                else :
                    scores_xgb.append(recall)
                    scores_xgb.append(acc)
                    scores_xgb.append(precision)
                    scores_xgb.append(f1)
                self.recalls.append(recall)
                self.accuracies.append(acc)
                self.precisions.append(precision)
                self.f1_macros.append(f1)
            print(f'Recall score for sample #{num}     (XGB)   =        {scores_xgb[0]}'.center(50))
            print(f'Precision score for sample #{num}  (XGB)   =        {scores_xgb[2]}'.center(50))
            print(f'F1 score for sample #{num}         (XGB)   =        {scores_xgb[3]}'.center(50))
            print(f'Accuracy score for sample #{num}   (XGB)   =        {scores_xgb[1]}'.center(50))
            print(f'Recall score for sample #{num}     (RF)    =        {scores_rf[0]}'.center(50))
            print(f'Precision score for sample #{num}  (RF)    =        {scores_rf[2]}'.center(50))
            print(f'F1 score for sample #{num}         (RF)    =        {scores_rf[3]}'.center(50))
            print(f'Accuracy score for sample #{num}   (RF)    =        {scores_rf[1]}'.center(50))
            print('-'*50)
            num += 1
    
    def validate_best_model(self):
        best_score_idx = self.recalls.index(max(self.recalls))
        x_test = pd.read_csv('/Users/salene/Documents/GitHub/driving-behavior-classification-prediction/etled-data/x_unseen.csv')
        y_test = pd.read_csv('/Users/salene/Documents/GitHub/driving-behavior-classification-prediction/etled-data/y_unseen.csv')
        y_test['Class'] = y_test['Class'].astype('category').cat.codes
        if best_score_idx != 0:
            model = self.models[1]
        else :
            model = self.models[0]
        print(f'Recall : {recall_score(model.predict(x_test), y_test, average="macro")}')
        print(f'Accuracy : {accuracy_score(model.predict(x_test), y_test)}')
        print(f'Precision : {precision_score(model.predict(x_test), y_test, average="macro")}')
        print(f'F1 : {f1_score(model.predict(x_test), y_test, average="macro")}')
    def get_results(self):
        return np.array(self.accuracies), np.array(self.recalls), np.array(self.f1_macros), np.array(self.precisions)

class MonteCarloSimulationWithStatisticalTests:
    def __init__(self, x, y, num_sims, sample_size):
        self.x = x
        self.y = y
        self.num_sims = num_sims
        self.sample_size = sample_size
    def run_simulation(self):
        original_class_probs = np.array([np.mean(self.y == c) for c in np.unique(self.y)])
        results = []
        for i in range(self.num_sims):
            simulated_sample_index = random.sample(range(len(self.x)), self.sample_size)
            simulated_X = self.x[simulated_sample_index, :]
            simulated_y = self.y[simulated_sample_index]
            simulated_class_probs = np.array([np.mean(simulated_y == c) for c in np.unique(self.y)])
            results.append(simulated_class_probs)
        return original_class_probs, results

if __name__ == '__main__':
    df = pd.read_csv('/Users/salene/Documents/GitHub/driving-behavior-classification-prediction/datasets/driving-behavior/train_motion_data.csv')
    target_col = 'Class'

    mcs = MonteCarloSimulation(df, target_col)
    mcs.run_simulation()
    results = mcs.get_results()
    print('Recalls:', results)
    print('Mean accuracy:', np.mean(results))
    print('Standard deviation:', np.std(results))
