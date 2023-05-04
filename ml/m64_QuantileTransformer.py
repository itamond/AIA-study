# 회귀로 맹그러
# 회귀데이터 올인 포문
# scaler 6개 올인

from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
# Quantile = 분위수. 정렬된 데이터를 특정 갯수로 나누는 기준이 되는 수
# QuantileTransformer :  정규분포로 나눈 다음 분위수로 나눈다. 스탠다드 스케일러와 민맥스 스케일러를 합친 느낌
# 분위수로 처리하기때문에 이상치에 자유롭다
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state=337, shuffle=True, stratify= y
)

scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = QuantileTransformer(n_quantiles=10)#분위수 지정 파라미터. 성능차이가 크다

# scaler = PowerTransformer(method='box-cox')
# scaler = PowerTransformer(method='yeo-johnson')

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델

model = RandomForestClassifier()

#3. 훈련

model.fit(x_train, y_train)

#4. 평가, 예측

print('결과 :', round(model.score(x_test,y_test),4))