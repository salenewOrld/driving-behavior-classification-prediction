import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
from config_reader import ConfigReader

''' More Graphes will be coming soon'''

class ReadData:
    def __init__(self, extension='csv', interpreter='Python'):
        self.extension = extension
        self.interpreter = interpreter
    def read_data(self, path):
        if self.interpreter == 'PySpark' and self.extension == 'csv':
            return spark.read.option('inferSchema', 'true').option('header', 'true').csv(path)
        elif self.interpreter == 'PySpark' and self.extension == 'parquet':
            return spark.read.option('inferSchema', 'true').option('header', 'true').parquet(path)
        elif self.interpreter == 'Python' and self.extension == 'csv':
            return pd.read_csv(path)
        elif self.interpreter == 'Python' and self.extension == 'parquet':
            return pd.read_parquet(path)
        return True
    
class Visualization(ReadData):
    def __init__(self, path, extension='csv', interpreter='Python', config=None):
        self.interpreter = interpreter     
        self.extension = extension
        self.config = config
        self.install_requirements()
        super().__init__(self.extension, self.interpreter)
        self.df = self.read_data(path)
        
    def install_requirements(self):
        if self.interpreter == 'Python':
            os.system('pip3 install seaborn')
            os.system('pip3 install matplotlib')
            os.system('pip3 install numpy')
        else :
            sc.install_pypi_package('seaborn')
            sc.install_pypi_package('matplotlib')
            sc.install_pypi_package('numpy')
        
    def histogram(self, param=None):
        if self.interpreter == 'Python':
            graph = sns.histplot(data=self.df, **param)
        if self.config != None and self.config['save'] == True:
            self.save(graph, self.config['output'])
        return graph
    def distribution(self, param=None, graph_type='violin'):
        if self.interpreter == 'Python' and graph_type == 'dist':
            graph = sns.distplot(**param)
        elif self.interpreter == 'Python' and graph_type == 'violin':
            graph = sns.violinplot(data=self.df, **param)
        if self.config != None and self.config['save'] == True:
            self.save(graph, self.config['output'])
        return graph
    def joint(self, param=None):
        if self.interpreter == 'Python':
            graph = sns.jointplot(data=self.df, **param)
        if self.config != None and self.config['save'] == True:
            self.save(graph, self.config['output'])
        return graph
    def pairplot(self, param=None):
        if param == None and self.interpreter == 'Python':
            graph = sns.pairplot(self.df)
        elif param != None and self.interpreter == 'Python': 
            graph = sns.pairplot(self.df, **param)
        if self.config != None and self.config['save'] == True:
            self.save(graph, self.config['output'])
        return graph
    def save(self, plot, output):
        plt.savefig(f'{output}/output.png')

def execute(config : dict):
    vis = Visualization(config['input_path'], config['extension'], config['interpreter'], config)
    param = config['param']
    if config['command'] == 'distribution':
        vis.distribution(config['param'])
    elif config['command'] == 'joint':
        vis.joint(param)
    elif config['command'] == 'pairplot':
        vis.pairplot(param)
    elif config['command'] == 'histogram':
        vis.histogram(param)

if __name__ == '__main__':
    try: 
        config = sys.argv[1]
        cfg = ConfigReader.read(config)
        execute(cfg)
    except:
        print('Missing config argument!')
    