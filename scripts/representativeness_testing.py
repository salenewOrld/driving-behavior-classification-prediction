import random
import numpy as np
from sklearn import linear_model, ensemble
from read_data import ReadData
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier

class RepresentativenessTesting:
    def __init__(self, method):
        self.method = method
    def execute(self, **param):
        # Monte Carlo and Predictive Modeling
        if self.method == 'mcapm':
            mcs = MonteCarloSimulation()
            results = mcs.monte_carlo_simulation(**param)
            pm = PredictiveModeling(results)
            return pm.predictive_modeling()
            
class MonteCarloSimulation:
    def __init__(self):
        pass

    def monte_carlo_simulation(self, data, model, features, target, iterations, test):
        results = []
        sss = StratifiedShuffleSplit(n_splits=iterations, test_size=0.2)
        for train_index, test_index in sss.split(data[features], data[target]):
            X_train, y_train = data.loc[train_index, features], data.loc[train_index, target]
            model.fit(X_train, y_train)
            results.append(model.predict(test[features]))
            
        return np.array(results)



    
class PredictiveModeling:
    def __init__(self, array_results):
        self.results = array_results
    def evaluate(self, model):
        testing_result = ''
        return testing_result
    def predictive_modeling(self):
        # example
        models_result = {}
        test_set = ''
        for j in len(self.results):
            model = linear_model.LinearRegression()
            model.fit(self.results[j].drop('y'), self.results[j]['y'])
            testing_result = self.evaluate(model)
            models_result[model] = testing_result
        return models_result
    
class DescriptiveSummaryStatsAnalysis(ReadData):
    def __init__(self, path, extension='csv', interpreter='Python'):
        self.path = path
        self.extension = extension
        self.interpreter = interpreter
        super().__init__(self.extension, self.interpreter)
        self.df = self.read_data(path)
    def descriptive(self, param=None):
        if param == None:
            return self.df.describe(percentiles=[.25, .5, .75], include='all', datetime_is_numeric=False)
        else :
            return self.df.describe(**param)
    def summary(self, param):
        df = self.df
        non_null = df[param['column']].count()
        print(f"| Summary statistical of {param['column']} |".center(50, '*'))
        print(f"| Mean of {param['column']} |".center(50, '-'), ':', df[param['column']].mean())
        print(f"| Median of {param['column']} |".center(50, '-'), ':', df[param['column']].median())
        print(f"| Standard deviation of {param['column']} |".center(50, '-'), ':', df[param['column']].std())
        print(f"| Variance of {param['column']} |".center(50, '-'), ':', df[param['column']].var())
        print(f"| Minimum value of {param['column']} |".center(50, '-'), ':', df[param['column']].min())
        print(f"| Maximum value of {param['column']} |".center(50, '-'), ':', df[param['column']].max())
        print(f"| Count of non-null values in {param['column']} |".center(50, '-'), ':', non_null)
        print(f"| Count of null values in {param['column']} |".center(50, '-'), ':', df[param['column']].shape[0] - non_null)
        print(f"| Sum of {param['column']} |".center(50, '-'), ':', df[param['column']].sum())
        print('\n')
    def execute(self):
        print(self.descriptive())
        for j in self.df.columns:
            if str(self.df[j].dtype) == 'object':
                pass
            else : 
                param = {'column' : j}
                self.summary(param)
        
    