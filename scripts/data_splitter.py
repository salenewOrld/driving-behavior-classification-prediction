from sklearn.model_selection import train_test_split
import pandas as pd

class Splitter:
    def __init__(self, x_path, test_path):
        self.x = pd.read_csv(x_path).drop(columns=['Class'])
        self.y = pd.read_csv(x_path)['Class']
        self.x_test = pd.read_csv(test_path).drop(columns=['Class'])
        self.y_test = pd.read_csv(test_path)['Class']
    
    def run(self, x, y, x_test, y_test):
        x_train, x_validate, y_train, y_validate = train_test_split(x, y, random_state=666, train_size=.75)
        x_train.to_csv('/Users/salene/Documents/GitHub/driving-behavior-classification-prediction/etled-data/x_train.csv', index=False)
        x_validate.to_csv('/Users/salene/Documents/GitHub/driving-behavior-classification-prediction/etled-data/x_unseen.csv', index=False)
        y_train.to_csv('/Users/salene/Documents/GitHub/driving-behavior-classification-prediction/etled-data/y_train.csv', index=False)
        y_validate.to_csv('/Users/salene/Documents/GitHub/driving-behavior-classification-prediction/etled-data/y_unseen.csv', index=False)
        self.x_test.to_csv('/Users/salene/Documents/GitHub/driving-behavior-classification-prediction/etled-data/x_test.csv', index=False)
        self.y_test.to_csv('/Users/salene/Documents/GitHub/driving-behavior-classification-prediction/etled-data/y_test.csv', index=False)
if __name__ == '__main__':
    splitter = Splitter(x_path = '/Users/salene/Documents/GitHub/driving-behavior-classification-prediction/datasets/driving-behavior/test_motion_data.csv', test_path='/Users/salene/Documents/GitHub/driving-behavior-classification-prediction/datasets/driving-behavior/test_motion_data.csv')
    splitter.run(splitter.x, splitter.y, splitter.x_test, splitter.y_test)