import random
import numpy as np
from sklearn import linear_model, ensemble
from read_data import ReadData
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
    def monte_carlo_simulation(data, n_simulations):
        # Define number of simulations
        n_simulations = n_simulations

        # Define the number of columns in the dataset
        n_cols = data.shape[1]

        # Define an empty array to store the simulation results
        results = np.zeros((n_simulations, n_cols))

        # Loop over the number of simulations
        for i in range(n_simulations):
            # Draw random samples from the data for each column
            for j in range(n_cols):
                results[i, j] = np.random.choice(data[:, j])

        # Return the results
        return results
    
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
        
    