# 10개 데이터셋
# 10개의 파일을 만든다
#[실습/과제] 피처를 한개씩 삭제하고 성능비교
#모델은 RF로만 한다


import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
#1. 데이터

x, y = load_diabetes(return_X_y=True)
x = pd.DataFrame(x).drop([1],axis=1)

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


# model.score : 0.4132272884510497
# r2 : 0.4132272884510497
# ====================================
# RandomForestRegressor() : [0.06868856 0.00840161 0.30556872 0.08390998 0.05098445 0.05069293
#  0.04960656 0.0178104  0.29993263 0.06440415]

# model.score : 0.4152212607124196
# r2 : 0.4152212607124196
# ====================================
# RandomForestRegressor() : [0.0726119  0.30394437 0.07771021 0.05475843 0.05383825 0.05102659
#  0.01788017 0.29924279 0.0689873 ]
