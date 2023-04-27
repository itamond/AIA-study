# 10개 데이터셋
# 10개의 파일을 만든다
#[실습/과제] 피처를 한개씩 삭제하고 성능비교
#모델은 RF로만 한다


import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
#1. 데이터

# x, y = load_iris(return_X_y=True)
x,y = load_iris(return_X_y=True)
x = pd.DataFrame(x).drop([0,1], axis = 1)
# x = pd.DataFrame(x).drop(0, axis=1)
# print(x.shape) #(150, 2)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, random_state=337
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = RandomForestClassifier()



#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score :', result)
y_predict = model.predict(x_test)
print('====================================')
print(model, ":", model.feature_importances_)

# RandomForestClassifier() : [0.12056223 0.02742518 0.44482851 0.40718408]



# model.score : 0.9666666666666667
# ====================================
# RandomForestClassifier() : [0.52541144 0.47458856]


