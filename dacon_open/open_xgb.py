import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split,GridSearchCV, KFold,RandomizedSearchCV
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
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
    
    # df = pd.get_dummies(df)
    
    
    
    return df

df = preprocessing(train)
test = preprocessing(test)

X = df.drop('Calories_Burned',axis = 1)
Y = df['Calories_Burned']

def objective(trial):
    # 하이퍼파라미터 탐색 범위
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.5),
        'subsample': trial.suggest_float('subsample', 0.5, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1),
        'gamma': trial.suggest_float('gamma', 0, 3),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300)
    }
    
    model = xgb.XGBRegressor(**param)
    
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
            model.fit(x_train, y_train)
            
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

# # study 객체 생성 및 실행
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=100)

# # 하이퍼파라미터 최적값 출력
# print(study.best_params())

# best_parms = {'n_estimators': 753, 'max_depth': 10, 'learning_rate': 0.13308388990265863, 
#                 'subsample': 0.9059989448554313, 'colsample_bytree': 0.7974906616951953, 'gamma': 1.6896272035459874, 
#                 'reg_alpha': 0.8845432907591179, 'reg_lambda': 0.810881097627782, 'min_child_weight': 2}

# model = XGBRegressor(**best_parms)

best_parms = {'n_estimators': 800, 'max_depth': 10, 'learning_rate': 0.1, 
                'subsample': 0.90, 'colsample_bytree': 0.8, 'gamma': 1.5, 
                'reg_alpha': 0.9, 'reg_lambda': 0.8, 'min_child_weight': 2}

model = XGBRegressor(**best_parms)

x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.2)

model.fit(x_train,y_train)

mean_squared_error(y_test,model.predict(x_test),
                    squared = False)

model.fit(X,Y)

model.predict(test)

sub = pd.read_csv('./_data/open/sample_submission.csv')

sub['Calories_Burned'] = model.predict(test)

sub = sub.set_index('ID')

sub.to_csv('./_save/open/optuna_3.csv')

model.fit(x_train,y_train)

model.predict(x_test)