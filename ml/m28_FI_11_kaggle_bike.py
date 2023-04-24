# [실습]
# 피처임포턴스가 전체 중요도에서 하위 20~25%인 컬럼들을 제거
# 재구성 후 모델을 돌려서 결과 도출

# 기존모델들과 성능비교

# 2. 모델구성
# model_list = [DecisionTreeClassifier(),RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]


import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
#1. 데이터

path='./_data/kaggle_bike/'
path_save='./_save/kaggle_bike/'    

train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0)                 

train_csv = train_csv.dropna()   

x = train_csv.drop(['count'], axis=1)   
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, random_state=337
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델

import matplotlib.pyplot as plt

model_list = [DecisionTreeRegressor(),RandomForestRegressor(), GradientBoostingRegressor(), XGBRegressor()]
model_name_list = ['DecisionTree', 'RandomForest', 'GradientDecentBoosting', 'XGBoost']


# def plot_feature_importances(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel('Feature Importances')
#     plt.ylabel('Features')
#     plt.ylim(-1, n_features)
#     plt.title(model)



# for i in range(4):
#     globals()['model'+str(i)] = model_list[i]
#     globals()['model'+str(i)].fit(x_train, y_train)
#     plt.subplot(2, 2, i+1)
#     # print(globals()['model'+str(i)].feature_importance_)
#     plot_feature_importances(globals()['model'+str(i)])
#     if i == 3:
#         plt.title('XGBClassifier()')
# plt.show()

for i, v in enumerate(model_list):
    model = v
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    r2= r2_score(y_pred, y_test)
    print(i+1,'.', model_name_list[i])
    print('기존 r2 :', r2)
    
    a = model.feature_importances_
    a = a.argmin(axis=0)
    
    x_d = pd.DataFrame(x).drop([a], axis=1)
    
    x_train_d, x_test_d, y_train_d, y_test_d = train_test_split(
        x_d, y, train_size=0.8, shuffle=True, random_state=337)
    
    x_train_d = scaler.fit_transform(x_train_d)
    x_test_d = scaler.transform(x_test_d)
    
    model.fit(x_train_d, y_train_d)
    result = model.score(x_test_d, y_test_d)
    print(f'{a}컬럼삭제 후 r2', result)


# 1 . DecisionTree
# 기존 r2 : 0.583492850909397
# 4컬럼삭제 후 r2 0.6076199842295863
# 2 . RandomForest
# 기존 r2 : 0.7358954068169159
# 3컬럼삭제 후 r2 0.802546852521786
# 3 . GradientDecentBoosting
# 기존 r2 : 0.7030301998693261
# 4컬럼삭제 후 r2 0.7801978457334504
# 4 . XGBoost
# 기존 r2 : 0.7957195038159817
# 3컬럼삭제 후 r2 0.8245402972987143