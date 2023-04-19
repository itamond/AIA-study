#컬런의 종류에 따라 훈련 결과에 악영향을 끼치는 불필요한 컬런이 있다.
#때문에 컬런을 걸러내는 작업을 함.
#ex)PCA

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#1. 데이터

x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, random_state=337
)

#2. 모델
model = DecisionTreeRegressor()
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
# model = XGBClassifier()
#트리 계열 모델들이다.


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score :', result)
y_predict = model.predict(x_test)
print('ACC :', accuracy_score(y_test, y_predict))

print('====================================')
print(model, ":", model.feature_importances_)






# iris : 컬럼4개 이에 대한 중요도를 출력한다. 최종 결과값을 도출하는데에 영향을 끼친 정도이다.

# ACC : 0.9666666666666667
# DecisionTreeClassifier() : [0.         0.01671193 0.40658454 0.57670353]

# ACC : 0.9666666666666667
# RandomForestClassifier() : [0.11696715 0.02794533 0.51764537 0.33744216]

# ACC : 0.9666666666666667
# GradientBoostingClassifier() : [0.00172841 0.01747272 0.71377823 0.26702064]

# ACC : 0.9666666666666667
# XGBClassifier() :[0.01794496 0.01218657 0.8486943  0.12117416]

# 1번 2번 컬런의 중요도가 낮아 삭제할 수 있다.
# 이럴경우 오히려 점수가 오르는 경우가 있다.