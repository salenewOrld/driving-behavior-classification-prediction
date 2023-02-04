from sklearn.model_selection import train_test_split
import pandas as pd

class Splitter:
    def __init__(self, x_path):
        self.x = pd.read_csv(x_path).drop(columns=['Class'])
        self.y = pd.read_csv(x_path)['Class']
    
    def run(self, x, y):
        x_train, x_validate, y_train, y_validate = train_test_split(x, y, random_state=666, train_size=.75)
        x_train.to_csv('/usr/src/etled-data/x_train.csv')
        x_validate.to_csv('/usr/src/etled-data/x_unseen.csv')
        y_train.to_csv('/usr/src/etled-data/y_train.csv')
        y_validate.to_csv('/usr/src/elted-data/y_unseen.csv')
        
if __name__ == '__main__':
    splitter = Splitter(x_path = '/usr/src/datasets/driving-behavior/train_motion_data.csv')
    