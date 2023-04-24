import pandas as pd
import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, HalvingRandomSearchCV, HalvingGridSearchCV
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.utils import all_estimators
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import all_estimators
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, Matern
import optuna
import datetime
import warnings
warnings.filterwarnings('ignore')

scaler_list = [MinMaxScaler(), MaxAbsScaler(), StandardScaler(), RobustScaler()]
model_list = [RandomForestRegressor(), DecisionTreeRegressor()]

def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))

path = './_data/open/'
path_save = './_save/open/'
path_save_min = './_save/open/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0).drop(['Weight_Status'], axis=1)
test_csv = pd.read_csv(path + 'test.csv', index_col=0).drop(['Weight_Status'], axis=1)
submit_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

x = train_csv.drop(['Calories_Burned'], axis=1)
y = train_csv['Calories_Burned']

x['Height(Feet)'] = 12*x['Height(Feet)']+x['Height(Remainder_Inches)']
x['Height(Remainder_Inches)'] = 703*x['Weight(lb)']/x['Height(Feet)']**2

test_csv['Height(Feet)'] = 12*test_csv['Height(Feet)']+test_csv['Height(Remainder_Inches)']
test_csv['Height(Remainder_Inches)'] = 703*test_csv['Weight(lb)']/test_csv['Height(Feet)']**2

le = LabelEncoder()
x['Gender'] = le.fit_transform(x['Gender'])
test_csv['Gender'] = le.transform(test_csv['Gender'])





min_rmse = 1

for k in range(1000000):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=k)

    for i in scaler_list:
        scaler = i
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        test_csv = scaler.transform(test_csv)

        def objective(trial, x_train, y_train, x_test, y_test, min_rmse):
            alpha = trial.suggest_loguniform('alpha', 0.0001, 1)
            n_restarts_optimizer  = trial.suggest_int('n_restarts_optimizer', 0, 5)
            # optimizer = trial.suggest_categorical('optimizer', ['Powell'])
            kernel_type = trial.suggest_categorical('kernel_type', ['rbf', 'matern', 'rq'])

            if kernel_type == 'rbf':
               kernel = RBF()
            elif kernel_type == 'matern':
                nu = trial.suggest_categorical('nu', [0.5, 1.5, 2.5])
                kernel = Matern(nu=nu)
            elif kernel_type == 'rq':
                alpha_rq = trial.suggest_uniform('alpha_rq', 0.01, 10.0)
                kernel = RationalQuadratic(alpha=alpha_rq)
            model = GaussianProcessRegressor(
                alpha=alpha,
                n_restarts_optimizer=n_restarts_optimizer,
                # optimizer=optimizer,

                kernel=kernel
            )
            
            model.fit(x_train, y_train)
            
            print('GPR result : ', model.score(x_test, y_test))
            
            y_pred = model.predict(x_test)
            rmse = RMSE(y_test, y_pred)
            print('GPR RMSE : ', rmse)
            if rmse < 0.308:
                submit_csv['Calories_Burned'] = model.predict(test_csv)
                date = datetime.datetime.now()
                date = date.strftime('%m%d_%H%M%S')
                submit_csv.to_csv(path_save + date + str(round(rmse, 5)) + '.csv')
                # if rmse < min_rmse:
                #     min_rmse = rmse
                #     submit_csv.to_csv(path_save_min + date + str(round(rmse, 5)) + '.csv')
            return rmse
        opt = optuna.create_study(direction='minimize')
        opt.optimize(lambda trial: objective(trial, x_train, y_train, x_test, y_test, min_rmse), n_trials=10000)
        print('best param : ', opt.best_params, 'best rmse : ', opt.best_value)
        