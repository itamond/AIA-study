# 10개 데이터셋
# 10개의 파일을 만든다
#[실습/과제] 피처를 한개씩 삭제하고 성능비교
#모델은 RF로만 한다


import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
#1. 데이터

x, y = load_wine(return_X_y=True)
x = pd.DataFrame(x).drop([0,1,2,3,4,5,7,8],axis=1)

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

# model.score : 0.9722222222222222
# ACC : 0.9722222222222222
# ====================================
# RandomForestClassifier() : [0.13638039 0.02621701 0.01677744 0.02190832 0.02984788 0.0617986
#  0.18357367 0.01359297 0.01827413 0.12909698 0.09773254 0.09222022
#  0.17257985]


# model.score : 0.9444444444444444
# ACC : 0.9444444444444444
# ====================================
# RandomForestClassifier() : [0.22034805 0.23881794 0.10285975 0.13938836 0.29858591]
