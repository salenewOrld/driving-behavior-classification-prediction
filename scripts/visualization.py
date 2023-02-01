import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
from config_reader import ConfigReader
import matplotlib.pyplot as plt
import scipy
from read_data import ReadData
import random
''' More Graphes will be coming soon'''

class Visualization(ReadData):
    def __init__(self, 
                 path, 
                 extension='csv', 
                 interpreter='Python', 
                 config=
        {
            'output' : '/usr/src/figs/',
            'save' : True
        }, 
                 save=False, 
                 output_path='/usr/src/figs/'):
        self.interpreter = interpreter     
        self.extension = extension
        self.config = config
        self.output_path = output_path
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
            self.save(graph, self.config['output'], 'hist')
        return graph
    def distribution(self, param=None, graph_type='violin'):
        if self.interpreter == 'Python' and graph_type == 'dist':
            graph = sns.distplot(**param)
        elif self.interpreter == 'Python' and graph_type == 'violin':
            graph = sns.violinplot(data=self.df, **param)
        if self.config != None and self.config['save'] == True:
            self.save(graph, self.config['output'], 'distribution')
        return graph
    def joint(self, param=None):
        if self.interpreter == 'Python':
            graph = sns.jointplot(data=self.df, **param)
        if self.config != None and self.config['save'] == True:
            self.save(graph, self.config['output'], 'joint')
        return graph
    def pairplot(self, param=None):
        if param == None and self.interpreter == 'Python':
            graph = sns.pairplot(self.df)
        elif param != None and self.interpreter == 'Python': 
            graph = sns.pairplot(self.df, **param)
        if self.config != None and self.config['save'] == True:
            self.save(graph, self.config['output'], 'pairplot')
        return graph
    def probplot(self, param=None):
        if self.interpreter == 'Python':
            graph = scipy.stats.probplot(plot=plt, **param)
        if self.config != None and self.config['save'] == True:
            self.save(graph, self.config['output'], 'probplot')
        return graph
    def save(self, plot, output, plot_name):
        plt.savefig(f'{output}/output_{plot_name}_{random.randint(0, 999999)}.png')

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
    