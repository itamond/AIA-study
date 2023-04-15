#all estimator =모든 모델에 대한 평가

import numpy as np
from sklearn.datasets import fetch_california_housing, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import all_estimators
import sklearn as sk

#1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, test_size=0.2
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
# model = RandomForestRegressor(n_estimators=100, n_jobs=4)  # n_estimators= epochs
allAlgorithms = all_estimators(type_filter='classifier')

# allAlgorithms = all_estimators(type_filter='classifier')


print('allAlgorithms :', allAlgorithms)
#튜플 안에 첫번째는 스트링 형태의 모델, 두번째는 클래스로 정의된 모델
max_r2 = 0
max_name = '바보'
for (name, algorithm) in allAlgorithms:
    try: #에러 예외처리
        model = algorithm()
        #3. 훈련
        model.fit(x_train, y_train)
        
        #4. 평가, 예측
        results = model.score(x_test, y_test)
        print(name,'의 정답률 :', results)
        if max_r2 < results :
            max_r2 = results
            max_name = name
        # y_predict = model.predict(x_test)
        # # print(y_test.dtype)  #데이터 타입 확인
        # # print(y_predict.dtype) #데이터 타입 확인    데이터 타입 변경은 astype 사용함.
        # r2 = r2_score(y_test, y_predict)
        # print('r2_score :', r2)
    except:
        #에러가 뜨면 except로 바로 넘어간다. 에러가 안뜨면 정상적으로 for문이 돌아감.
        print(name, '은(는) 에러뜬 놈!!!')
        #에러가 뜨는 모델들은 기본적으로 파라미터 수정이 필요한 모델들이다.
print('===================================')
print('최고모델 :', max_name, max_r2)
print('===================================')

# 최고모델 : ExtraTreesClassifier 0.9833333333333333