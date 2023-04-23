import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split,GridSearchCV, KFold,RandomizedSearchCV
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
import lightgbm as lgb
import xgboost as xgb
import optuna
import warnings
warnings.filterwarnings(action = 'ignore')


train = pd.read_csv('./_data/open/train.csv')
test = pd.read_csv('./_data/open/test.csv')

def min_max_scaling(x):
    return (x - np.min(x)) / (max(x) - min(x))

def scaler(df):
    
    cols = df.describe().columns
    
    for col in cols:
        
        if col != 'Calories_Burned':
            df[col] = min_max_scaling(df[col])
        
    return df

def preprocessing(df):
    
    df['height'] = df['Height(Feet)'] + df['Height(Remainder_Inches)'] * 0.12
        
    df = scaler(df)
    df = df.drop(['ID','Height(Feet)','Height(Remainder_Inches)'], axis = 1)
    
    df = pd.get_dummies(df)
    
    
    
    return df

df = preprocessing(train)
test = preprocessing(test)

X = df.drop('Calories_Burned',axis = 1)
Y = df['Calories_Burned']

def objective(trial):
    # 하이퍼파라미터 탐색 범위
    param = {
        'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.5),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'boosting':'dart',
        'objective':'regression',        
        'metric':'mse',
        'is_traing_metric':True,
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1),
        'num_leaves': trial.suggest_int('num_leaves', 140, 300),
        'bagging_freq': trial.suggest_int('bagging_freq', 3, 10)
    }
    
    model = lgb.train(param, lgb.Dataset(X, label=Y), 1000, verbose_eval=100, early_stopping_rounds=100)

    
    # 모델 정의
    
    
    valid_cv = KFold(n_splits = 5,
                shuffle = True)
    
    
    i = 0
    
    result_rmse = []
    
    while i !=5:
        
        x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.2)
        
        
        mse_list = []
        
        for train_idx, valid_idx in valid_cv.split(x_train):
            
            train_x , test_x = x_train.iloc[train_idx], y_train.iloc[train_idx]
            valid_x , valid_y = x_train.iloc[valid_idx], y_train.iloc[valid_idx]
            
            # 모델 학습
            # model.fit(x_train, y_train)
            
            # 검증 데이터에 대한 예측값 계산
            y_pred = model.predict(valid_x)
            
            # 검증 데이터에 대한 예측값과 실제값 사이의 평균 제곱 오차 계산
            mse = mean_squared_error(valid_y, y_pred,
                                        squared = False)
            
            mse_list.append(mse)
            
        i += 1
        
        result_rmse.append(np.mean(mse_list))
                    
    # 목적 함수 반환값
    return np.mean(result_rmse)

# study 객체 생성 및 실행
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=100)

# 하이퍼파라미터 최적값 출력
# print(study.best_params())
# [I 2023-04-21 17:57:27,692] Trial 99 finished with value: 0.13677043774216896 and parameters: {'learning_rate': 0.4781444056865173,
#                                                                                                'max_depth': 9, 'bagging_fraction': 0.9467654704551626,
#                                                                                                'feature_fraction': 0.8209196899429982, 'num_leaves': 289
#                                                                                                'bagging_freq': 9}. Best is trial 72 with value: 0.11700291475042876.




x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.2)

# model.fit(x_train,y_train)

# mean_squared_error(y_test,model.predict(x_test),
#                     squared = False)

# model.fit(X,Y)

# model.predict(test)

train_ds = lgb.Dataset(x_train, label= y_train)
test_ds = lgb.Dataset(x_test, label = y_test)
params = {'learning_rate': 0.3,
          'max_depth': 9,
          'boosting':'gbdt',
          'objective':'regression',
          'metric':'mse',
          'is_traing_metric':True,
          'num_leaves':289,
          'feature_fraction':0.82,
          'bagging_fraction':0.94,
          'bagging_freq':9,
          'seed':2020}
# 러닝0.47
# 맥스뎁스 9
# 배깅 프랙션 0.94
# 피쳐 프랙션 0.82
# 넘 리브스 289
# 배깅 프렉 9
model = lgb.train(params, train_ds, 1000, test_ds, verbose_eval=100, early_stopping_rounds=100)

sub = pd.read_csv('./_data/open/sample_submission.csv')

sub['Calories_Burned'] = model.predict(test)

sub = sub.set_index('ID')

sub.to_csv('./_save/open/optuna_4.csv')

# # model.fit(x_train,y_train)

# # model.predict(x_test)