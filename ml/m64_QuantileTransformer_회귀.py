import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.metrics import r2_score

#1 데이터
ddarung_path = 'C:/AIA/AIA-study/_data/ddarung/'
kaggle_bike_path = 'C:/AIA/AIA-study/_data/kaggle_bike/'

ddarung = pd.read_csv(ddarung_path + 'train.csv', index_col = 0).dropna()
kaggle_bike = pd.read_csv(kaggle_bike_path + 'train.csv', index_col = 0).dropna()

x1 = ddarung.drop(['count'], axis = 1)
y1 = ddarung['count']

x2 = kaggle_bike.drop(['count', 'casual', 'registered'], axis = 1)
y2 = kaggle_bike['count']

data_list = {'ddarung' : (x1, y1),
             'kaggle_bike' : (x2, y2),
             'california' : fetch_california_housing,
             'diabetes' : load_diabetes}

scaler_list = {'MinMax' : MinMaxScaler(),
               'Max' : MaxAbsScaler(),
               'Standard' : StandardScaler(),
               'Robust' : RobustScaler(),
               'QuantileTransformer' : QuantileTransformer(n_quantiles=10),
               'PowerTransformer' : PowerTransformer(method='yeo-johnson')}

        
for d in data_list:
    max_r2 = -1
    max_scaler = None
    if d == 'ddarung' or d == 'kaggle_bike':
        x, y = data_list[d]
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 1234)
    elif d == 'california':
        x, y = data_list[d](return_X_y = True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 1234)
    elif d == 'diabetes':
        x, y = data_list[d](return_X_y = True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 1234)

    for s in scaler_list:
        scaler = scaler_list[s]
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        model = RandomForestRegressor()
        model.fit(x_train_scaled, y_train)
        y_predict = model.predict(x_test_scaled)
        r2 = np.round(r2_score(y_test, y_predict),2)
        if r2 > max_r2:
            max_r2 = r2
            max_scaler = s
        print(f'데이터 : {d}, 스케일러 : {s}, 결정 계수 : {r2}')
    print(f'데이터 : {d}, 가장 높은 결정 계수 : {max_r2}, 가장 높은 결정 계수 스케일러 : {max_scaler}')
    
    
# 데이터 : ddarung, 스케일러 : MinMax, 결정 계수 : 0.79
# 데이터 : ddarung, 스케일러 : Max, 결정 계수 : 0.8
# 데이터 : ddarung, 스케일러 : Standard, 결정 계수 : 0.79
# 데이터 : ddarung, 스케일러 : Robust, 결정 계수 : 0.8
# 데이터 : ddarung, 스케일러 : QuantileTransformer, 결정 계수 : 0.79
# 데이터 : ddarung, 스케일러 : PowerTransformer, 결정 계수 : 0.8
# 데이터 : ddarung, 가장 높은 결정 계수 : 0.8, 가장 높은 결정 계수 스케일러 : Max
# 데이터 : kaggle_bike, 스케일러 : MinMax, 결정 계수 : 0.3
# 데이터 : kaggle_bike, 스케일러 : Max, 결정 계수 : 0.3
# 데이터 : kaggle_bike, 스케일러 : Standard, 결정 계수 : 0.3
# 데이터 : kaggle_bike, 스케일러 : Robust, 결정 계수 : 0.3
# 데이터 : kaggle_bike, 스케일러 : QuantileTransformer, 결정 계수 : 0.3
# 데이터 : kaggle_bike, 스케일러 : PowerTransformer, 결정 계수 : 0.3
# 데이터 : kaggle_bike, 가장 높은 결정 계수 : 0.3, 가장 높은 결정 계수 스케일러 : MinMax
# 데이터 : california, 스케일러 : MinMax, 결정 계수 : 0.8
# 데이터 : california, 스케일러 : Max, 결정 계수 : 0.8
# 데이터 : california, 스케일러 : Standard, 결정 계수 : 0.8
# 데이터 : california, 스케일러 : Robust, 결정 계수 : 0.8
# 데이터 : california, 스케일러 : QuantileTransformer, 결정 계수 : 0.8
# 데이터 : california, 스케일러 : PowerTransformer, 결정 계수 : 0.72
# 데이터 : california, 가장 높은 결정 계수 : 0.8, 가장 높은 결정 계수 스케일러 : MinMax
# 데이터 : diabetes, 스케일러 : MinMax, 결정 계수 : 0.46
# 데이터 : diabetes, 스케일러 : Max, 결정 계수 : 0.45
# 데이터 : diabetes, 스케일러 : Standard, 결정 계수 : 0.48
# 데이터 : diabetes, 스케일러 : Robust, 결정 계수 : 0.46
# 데이터 : diabetes, 스케일러 : QuantileTransformer, 결정 계수 : 0.48
# 데이터 : diabetes, 스케일러 : PowerTransformer, 결정 계수 : 0.45
# 데이터 : diabetes, 가장 높은 결정 계수 : 0.48, 가장 높은 결정 계수 스케일러 : Standard