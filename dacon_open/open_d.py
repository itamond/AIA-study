import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, MaxAbsScaler, RobustScaler, StandardScaler, Normalizer
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import PolynomialFeatures
import optuna
import datetime
import warnings
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
warnings.filterwarnings('ignore')
from scipy import optimize
from skopt import BayesSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score
from skopt.space import Real, Integer
from skopt.sampler import Lhs

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

x['Height(inch)'] = 12*x['Height(Feet)']+x['Height(Remainder_Inches)']
test_csv['Height(inch)'] = 12*test_csv['Height(Feet)']+test_csv['Height(Remainder_Inches)']

# x['BMI'] = np.round((703*x['Weight(lb)']/x['Height(Feet)']**2),2)
# test_csv['BMI'] = np.round((703*test_csv['Weight(lb)']/test_csv['Height(Feet)']**2),2)

le = LabelEncoder()
x['Gender'] = le.fit_transform(x['Gender'])
test_csv['Gender'] = le.transform(test_csv['Gender'])

x = x.drop(['Height(Feet)', 'Height(Remainder_Inches)'], axis=1)
test_csv = test_csv.drop(['Height(Feet)', 'Height(Remainder_Inches)'], axis=1)

poly = PolynomialFeatures(degree=1,interaction_only=True, include_bias=True)



# scaler2 = StandardScaler()
# x = pd.DataFrame(scaler2.fit_transform(x))
# test_csv = pd.DataFrame(scaler2.transform(test_csv))

# scaler = MinMaxScaler()
# x = pd.DataFrame(scaler.fit_transform(x))
# test_csv = pd.DataFrame(scaler.transform(test_csv))

scaler = MaxAbsScaler()
x = pd.DataFrame(scaler.fit_transform(x))
test_csv = pd.DataFrame(scaler.transform(test_csv))


x = pd.DataFrame(poly.fit_transform(x))
test_csv = pd.DataFrame(poly.transform(test_csv))


for i in range(10000):
    kf = KFold(n_splits=5, shuffle=True, random_state=i+128)
#     _, x_test, _, y_test = train_test_split(x, y, test_size=0.2, random_state=128, shuffle=True)
    for train_idx, test_idx in kf.split(x):
       x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
       y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    def objective(trial):
       alpha = trial.suggest_loguniform('alpha',  0.000000001, 0.1)
       n_restarts_optimizer  = trial.suggest_int('n_restarts_optimizer', 5, 60)
       optimizer = trial.suggest_categorical('optimizer', ['fmin_l_bfgs_b', 'Powell', 'CG'])         
       model = GaussianProcessRegressor(
           alpha=alpha,
           n_restarts_optimizer=n_restarts_optimizer,
           optimizer=optimizer,   
           # optimizer=my_optimizer,       
       )     
   
       model.fit(x, y)
   
       y_pred = np.round(model.predict(x_test))
       rmse = RMSE(y_test, y_pred)
       print('GPR RMSE : ', rmse)
       if rmse < 0.15:
           submit_csv['Calories_Burned'] = np.round(model.predict(test_csv))
           date = datetime.datetime.now()
           date = date.strftime('%m%d_%H%M%S')
           submit_csv.to_csv(path_save +str(round(rmse, 4)) + '.csv')
       return rmse
   
   
    sampler = TPESampler(
        seed=42,
    #     n_startup_trials=5,
    #     gamma=0.5,
    #     weights={"learning_rate": 1.0, "num_layers": 0.5}
    )
    # sampler = Lhs()
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
    # pruner = SuccessiveHalvingPruner(
    #     min_resource=1,  # 최소 자원
    #     reduction_factor=2,  # 감소 비율
    # #     n_warmup_steps=10  # Pruning을 적용하기 전의 조기 중지 횟수
    # )       
    opt = optuna.create_study(
        study_name='regression_study', 
        sampler=sampler, 
        pruner=pruner, 
        direction='minimize'
    )
    # opt = optuna.create_study(direction='minimize')
    opt.optimize(objective, n_trials=200)
    print('best param : ', opt.best_params, 'best rmse : ', opt.best_value)
   