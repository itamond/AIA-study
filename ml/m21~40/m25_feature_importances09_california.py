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
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
#1. 데이터

x, y = fetch_california_housing(return_X_y=True)
x = pd.DataFrame(x).drop([1,2,3,4],axis=1)
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


# model.score : 0.8010711577471686
# r2 : 0.8010711577471686
# ====================================
# RandomForestRegressor() : [0.52473256 0.05336135 0.04765666 0.03042121 0.0309526  0.13490861
#  0.08837886 0.08958815]

# model.score : 0.8042336436742961
# r2 : 0.8042336436742961
# ====================================
# RandomForestRegressor() : [0.53494679 0.15845942 0.14898942 0.15760437]
