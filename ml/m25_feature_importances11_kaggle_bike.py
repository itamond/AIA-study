# 10개 데이터셋
# 10개의 파일을 만든다
#[실습/과제] 피처를 한개씩 삭제하고 성능비교
#모델은 RF로만 한다


import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,r2_score
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

#2. 모델
model = RandomForestRegressor()



#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score :', result)
y_predict = model.predict(x_test)
print('r2 :', r2_score(y_test, y_predict))

print('====================================')
print(model, ":", model.feature_importances_)



# model.score : 0.9997311721904149
# r2 : 0.9997311721904149
# ====================================
# RandomForestRegressor() : [2.68741659e-05 2.54399204e-06 1.06053315e-05 1.25961272e-05
#  5.82404614e-05 6.01498157e-05 7.77849654e-05 6.67780034e-05
#  4.93620566e-02 9.50322371e-01]

