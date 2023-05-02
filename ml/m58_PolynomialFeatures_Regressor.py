import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.metrics import r2_score

#1 데이터
ddarung_path = 'c:/AIA/AIA-study/_data/ddarung/'
kaggle_bike_path = 'c:/AIA/AIA-study/_data/kaggle_bike/'

ddarung = pd.read_csv(ddarung_path + 'train.csv', index_col = 0).dropna()
kaggle_bike = pd.read_csv(kaggle_bike_path + 'train.csv', index_col = 0).dropna()

x1 = ddarung.drop(['count'], axis = 1)
y1 = ddarung['count']

x2 = kaggle_bike.drop(['count', 'casual', 'registered'], axis = 1)
y2 = kaggle_bike['count']

data_list = {'ddarung' : (x1, y1),
             'kaggle_bike' : (x2, y2),
             'california' : fetch_california_housing}

scaler_list = {'MinMax' : MinMaxScaler(),
               'Max' : MaxAbsScaler(),
               'Standard' : StandardScaler(),
               'Robust' : RobustScaler()}

model_list = [XGBRegressor(),
              LGBMRegressor(),
              RandomForestRegressor(),
              DecisionTreeRegressor(),
              CatBoostRegressor(verbose = 0)]

pf = PolynomialFeatures()

for d in data_list:
    if d == 'ddarung' or d == 'kaggle_bike':
        x, y = data_list[d]
        x = pf.fit_transform(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 1234)
    elif d == 'california':
        x, y = data_list[d](return_X_y = True)
        x = pf.fit_transform(x)        
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 1234)
    for s in scaler_list:
        scaler = scaler_list[s]
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        models = [('xgb', model_list[0]), ('lgbm', model_list[1]), ('rf', model_list[2]), ('dt', model_list[3]), ('cat', model_list[4])]
        model = StackingRegressor(estimators = models)
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        r2 = r2_score(y_test, y_predict)
        print(f'데이터 : {d}, 스케일러 : {s}, 결정 계수 : {r2}')