#컬런의 종류에 따라 훈련 결과에 악영향을 끼치는 불필요한 컬런이 있다.
#때문에 컬런을 걸러내는 작업을 함.
#ex)PCA

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score

#1. 데이터

x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, random_state=337
)

#2. 모델
# model = DecisionTreeRegressor()
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
model = XGBRegressor()
#트리 계열 모델들이다.


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score :', result)
y_predict = model.predict(x_test)
print('====================================')
print(model, ":", model.feature_importances_)






# iris : 컬럼4개 이에 대한 중요도를 출력한다. 최종 결과값을 도출하는데에 영향을 끼친 정도이다.

# DecisionTreeRegressor() : [0.06662522 0.00167182 0.26110973 0.07622547 0.0557328  0.05771134
#  0.03058018 0.01270558 0.35441937 0.08321849]
# RandomForestRegressor() : [0.072765   0.00856924 0.28496008 0.07702124 0.05138937 0.05237631
#  0.04682154 0.01751792 0.32099232 0.06758698]
# GradientBoostingRegressor() : [0.06480998 0.00881152 0.24778406 0.08409326 0.04055535 0.0390411
#  0.03921319 0.00817009 0.40481181 0.06270965]
# XGBRegressor() : [0.0330486  0.05173958 0.1607633  0.0750218  0.05887739 0.04195798
#  0.05592556 0.03102507 0.43406495 0.05757575]