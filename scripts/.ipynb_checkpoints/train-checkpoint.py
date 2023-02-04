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
from catboost import CatBoostRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn import metrics 

class ModelsBenchmark:
    def __init__(self) -> None:
        self.models = None
        self.cfg = None
    def get_models(self, problem_type) -> list:
        models = {
            'classification' : 
            {
                "models" : [MLPClassifier(), KNeighborsClassifier(), SVC(), GaussianProcessClassifier(), RBF(), DecisionTreeClassifier(), GaussianNB(), RandomForestClassifier(), AdaBoostClassifier()],
                'classifier_name' : ['MLPClassifier', 'KNeightborsClassifier', 'SVC', 'GaussianProcessClassifier', 'RBF', 'DecisionTreeClassifier', 'GaussianNB', 'RandomForestClassifier', 'AdaBoostClassifier']
            },
            'regression' : 
            {
                'models' : [LinearRegression(), ElasticNet(), SGDRegressor(), BayesianRidge(), RandomForestRegressor(), GradientBoostingRegressor(), KernelRidge(), CatBoostRegressor(), XGBRegressor(), LGBMRegressor(), DecisionTreeRegressor()],
                'regressor_name' : ['LinearRegression', 'ElasticNet', 'SGDRegressor', 'BayesianRidge', 'RandomForestRegressor', 'GradientBoostingRegressor', 'KernelRidge', 'CatBoostRegressor', 'XGBRegressor', 'LGBMRegressor', 'DecisionTreeRegressor']
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
            'y_train' : pd.read_csv(config['y_train_path']),
            'x_test' : pd.read_csv(config['x_test_path']),
            'y_test' : pd.read_csv(config['y_test_path']),
            'x_validate' : pd.read_csv(config['x_validate_path']),
            'y_validate' : pd.read_csv(config['y_validate_path'])
        }
        return data
    #def get_best_model(self, models, metrics):
        
    def evaluate(self, models, x_test, y_test) -> dict:
        result = {}
        if self.cfg['problem_type'] == 'classification':
            for index, value in enumerate(models['models']):
                result[models['classifier_name'][index]]['precision'] = metrics.precision_score(value.predict(x_test), y_test)
                result[models['classifier_name'][index]]['f1_score'] = metrics.f1_score(value.predict(x_test), y_test)
                result[models['classifier_name'][index]]['recall'] = metrics.recall_score(value.predict(x_test), y_test)
                result[models['classifier_name'][index]]['roc_auc'] = metrics.roc_auc_score(value.predict(x_test), y_test)
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
        return result
    def execute(self, config_path : str):
        self.cfg : dict = cr.read(config_path)
        self.problem_type : str = self.cfg['problem_type']
        self.models : list = self.get_models(self.problem_type)
        data : dict = self.get_x_y(self.cfg)
        models_trained : list = self.fit(self.models, data['x_train'], data['y_train'])
        evaluated : dict = self.evaluate(models_trained, data['x_test'], data['y_test'])
        #evaludated_with_unseen : dict = self.evaluate()