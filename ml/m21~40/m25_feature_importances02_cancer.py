# 10개 데이터셋
# 10개의 파일을 만든다
#[실습/과제] 피처를 한개씩 삭제하고 성능비교
#모델은 RF로만 한다


import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
#1. 데이터

x, y = load_breast_cancer(return_X_y=True)

x = pd.DataFrame(x).drop([5,9,10,11,12,13,15,16,17,19,20,29], axis=1)

print(x.shape)
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, random_state=337
)

#2. 모델
model = RandomForestClassifier()



#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score :', result)
y_predict = model.predict(x_test)
print('ACC :', accuracy_score(y_test, y_predict))

print('====================================')
print(model, ":", model.feature_importances_)
# model.score : 0.9649122807017544
# ACC : 0.9649122807017544
# ====================================
# RandomForestClassifier() : [0.04542996 0.0163454  0.04546404 0.03938134 0.00630104 0.01661706
#  0.02731226 0.13528272 0.00345095 0.00415013 0.00662412 0.00513143
#  0.00358949 0.03666664 0.00399963 0.00228085 0.00811728 0.011604
#  0.00394649 0.00619548 0.09541092 0.01838399 0.18806783 0.11581476
#  0.02246265 0.00979734 0.02766615 0.07811764 0.00833372 0.0080547 ]



# model.score : 0.9649122807017544
# ACC : 0.9649122807017544
# ====================================
# RandomForestClassifier() : [0.03432931 0.0194447  0.05096853 0.06103193 0.00728461 0.06503485
#  0.1019143  0.00475726 0.00569169 0.00627609 0.02163905 0.1950832
#  0.188078   0.02152264 0.01273249 0.04029811 0.15097069 0.01294257]
