# 10개 데이터셋
# 10개의 파일을 만든다
#[실습/과제] 피처를 한개씩 삭제하고 성능비교
#모델은 RF로만 한다


import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#1. 데이터

x, y = fetch_covtype(return_X_y=True)

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





