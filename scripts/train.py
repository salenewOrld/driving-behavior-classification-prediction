from config_reader import ConfigReader as cr
import visualization as v
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.kernel_ridge import KernelRidge
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost.sklearn import XGBRegressor, XGBClassifier
#from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn import metrics 
import pandas as pd

class ModelsBenchmark:
    def __init__(self, config_path) -> None:
        self.models = None
        self.cfg = cr.read(config_path)
    def get_models(self, problem_type) -> list:
        models = {
            'classification' : 
            {
                "models" : [MLPClassifier(), KNeighborsClassifier(), SVC(), GaussianProcessClassifier(), DecisionTreeClassifier(), GaussianNB(), RandomForestClassifier(), AdaBoostClassifier(), CatBoostClassifier(), XGBClassifier()],
                'classifier_name' : ['MLPClassifier', 'KNeightborsClassifier', 'SVC', 'GaussianProcessClassifier', 'DecisionTreeClassifier', 'GaussianNB', 'RandomForestClassifier', 'AdaBoostClassifier', 'CatBoostClassifier', 'XGBClassifier']
            },
            'regression' : 
            {
                'models' : [LinearRegression(), ElasticNet(), SGDRegressor(), BayesianRidge(), RandomForestRegressor(), GradientBoostingRegressor(), KernelRidge(), CatBoostRegressor(), XGBRegressor(), DecisionTreeRegressor()],
                'regressor_name' : ['LinearRegression', 'ElasticNet', 'SGDRegressor', 'BayesianRidge', 'RandomForestRegressor', 'GradientBoostingRegressor', 'KernelRidge', 'CatBoostRegressor', 'XGBRegressor', 'DecisionTreeRegressor']
            }
        }
        return models[problem_type]
    def fit(self, models, x_train, y_train) -> list:
        for j in range(len(models['models'])):
            models['models'][j].fit(x_train, y_train)
        return models
    def get_x_y(self, config):
        data = {
            'x_train' : pd.read_csv(config['x_train_path']),
            'y_train' : pd.read_csv(config['y_train_path'])['Class'].astype('category').cat.codes,
            'x_test' : pd.read_csv(config['x_test_path']),
            'y_test' : pd.read_csv(config['y_test_path'])['Class'].astype('category').cat.codes,
            'x_validate' : pd.read_csv(config['x_validate_path']),
            'y_validate' : pd.read_csv(config['y_validate_path'])['Class'].astype('category').cat.codes
        }
        return data
    #def get_best_model(self, models, metrics):
        
    def evaluate(self, models, x_test, y_test) -> dict:
        if self.cfg['problem_type'] == 'classification':
            models_arr = []
            models_name = []
            precision = []
            f1s = []
            recall = []
            #roc_auc = []
            for index, value in enumerate(models['models']):
                models_arr.append(value)
                models_name.append(models['classifier_name'][index])
                precision.append(metrics.precision_score(value.predict(x_test), y_test, average='macro'))
                f1s.append(metrics.f1_score(value.predict(x_test), y_test, average='macro'))
                recall.append(metrics.recall_score(value.predict(x_test), y_test, average='macro'))
                #roc_auc.append(metrics.roc_auc_score(value.predict(x_test), y_test))
        elif self.cfg['problem_type'] == 'regression':
            for index, value in enumerate(models['models']):
                result[models['regressor_name'][index]]['explained_variance'] = metrics.explained_variance_score(value.predict(x_test), y_test)
                result[models['regressor_name'][index]]['max_error'] = metrics.max_error(value.predict(x_test), y_test)
                result[models['regressor_name'][index]]['mean_absolute_score'] = metrics.mean_absolute_error(value.predict(x_test), y_test)
                result[models['regressor_name'][index]]['mean_squared_error'] = metrics.mean_squared_error(value.predict(x_test), y_test, root=True)
                result[models['regressor_name'][index]]['root_mean_squared_error'] = metrics.mean_squared_error(value.predict(x_test), y_test, root=False)
                result[models['regressor_name'][index]]['mean_squared_log_error'] = metrics.mean_squared_log_error(value.predict(x_test), y_test)
                result[models['regressor_name'][index]]['median_absolute_error'] = metrics.mean_squared_log_error(value.predict(x_test), y_test)
                result[models['regressor_name'][index]]['r2'] = metrics.r2_score(value.predict(x_test), y_test)
                result[models['regressor_name'][index]]['median_absolute_error'] = metrics.mean_squared_log_error(value.predict(x_test), y_test)
                result[models['regressor_name'][index]]['mean_poisson_deviance'] = metrics.mean_poisson_deviance(value.predict(x_test), y_test)
        return pd.DataFrame({
            "model_name" : models_name,
            "precision" : precision,
            'f1' : f1s,
            'recall' : recall,
            #'roc_auc' : roc_auc
        }), models_arr
    def validate_best_model(self, model, model_name,x_validate, y_validate):
        return pd.DataFrame({
            'model_name' : [model_name],
            'precision' : [metrics.precision_score(model.predict(x_validate), y_validate, average='macro')],
            'f1' : [metrics.f1_score(model.predict(x_validate), y_validate, average='macro')],
            'recall' : [metrics.recall_score(model.predict(x_validate), y_validate, average='macro')],
        })
    def execute(self, config_path : str, export=False):
        self.cfg : dict = cr.read(config_path)
        self.problem_type : str = self.cfg['problem_type']
        self.models : list = self.get_models(self.problem_type)
        data : dict = self.get_x_y(self.cfg)
        models_trained : list = self.fit(self.models, data['x_train'], data['y_train'])
        evaluated, models_arr = self.evaluate(models_trained, data['x_test'], data['y_test'])
        best_model = models_arr[evaluated['recall'].to_list().index(max(evaluated['recall'].to_list()))]
        best_model_name = evaluated['model_name'][evaluated['recall'].to_list().index(max(evaluated['recall'].to_list()))]
        evaluated_with_unseen = self.validate_best_model(best_model, best_model_name, data['x_validate'], data['y_validate'])
        if export:
            evaluated.to_excel('/Users/salene/Documents/GitHub/driving-behavior-classification-prediction/results/driving_behavior_model_performance.xlsx', sheet_name='Overall Performance')
        return evaluated_with_unseen, models_arr

if __name__ == '__main__':
    cfg_path = '/Users/salene/Documents/GitHub/driving-behavior-classification-prediction/configs/experiment-1.yaml'
    trainer = ModelsBenchmark(cfg_path)
    result, models_arr = trainer.execute(cfg_path, export=True)
    print(result)