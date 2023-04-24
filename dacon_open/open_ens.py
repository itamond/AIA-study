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
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
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
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, shuffle=True, random_state=337)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)
poly = PolynomialFeatures()
x_train = pd.DataFrame(poly.fit_transform(x_train))
x_test = pd.DataFrame(poly.fit_transform(x_test))
test_csv =  pd.DataFrame(poly.transform(test_csv))

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state = 777)

# best_parms = {'n_estimators': 800, 'max_depth': 10, 'learning_rate': 0.1, 
#                 'subsample': 0.90, 'colsample_bytree': 0.8, 'gamma': 1.5, 
#                 'reg_alpha': 0.9, 'reg_lambda': 0.8, 'min_child_weight': 2}

# xgb_model = XGBRegressor(**best_parms)

poly = PolynomialFeatures(degree=2, include_bias=False)
x_train = poly.fit_transform(x_train)
x_test = poly.transform(x_test)

mlp_model = MLPRegressor(hidden_layer_sizes=(1024, 512,2), max_iter=500, activation='relu', solver='adam', random_state=42)

gau_model = GaussianProcessRegressor(
    alpha=0.0001237,
    n_restarts_optimizer=4,
    optimizer='Powell',)

estimators = [('mlp', mlp_model), ('gau', gau_model)]
model = StackingRegressor(estimators=estimators, cv=kfold, final_estimator=gau_model)


model.fit(x_train, y_train)


print('GPR result : ', model.score(x_test, y_test))


y_pred = model.predict(x_test)
rmse = RMSE(y_test, y_pred)
print('GPR RMSE : ', rmse)


submit_csv['Calories_Burned'] = model.predict(test_csv)
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')
submit_csv.to_csv(path_save + date + str(round(rmse, 5)) + '.csv')
