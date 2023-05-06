#분류 싹 모아서 테스트

import numpy as np
from sklearn.datasets import fetch_covtype, load_iris, load_breast_cancer, load_wine, load_digits
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, RobustScaler

#1. 데이터
# datasets = load_iris()
# x = datasets.data
# y = datasets['target']


#2. 모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
#랜덤 포레스트는 sklearn의 앙상블에 있다. 랜덤 포레스트는 DecisionTree가 앙상블된 모델이다.




#모델의 classifier 분류모델 regressor 회귀모델.
#문제에 따라 다른 모델씀
#서포트 백터 머신

# model = Sequential()
# model.add(Dense(10, activation='relu', input_shape=(4,)))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# # model.add(Dense(3, activation='softmax'))
# # model = LinearSVC()  #알고리즘 연산이 다 포함되어있다. 나중에 다 공부해라. 어떤 모델인지 정도는 알아야한다.
# model = RandomForestRegressor()


# #LinearSVC의 파라미터 C 는 작으면 작을수록 직선을 긋는다.
# #분류문제에서 딥러닝은 layer를 거쳐 선을 수정하지만, 머신러닝은 단층 레이어로 선을 그어 클래스를 분류한다.
# #때문에 고도화된 문제는 완벽한 분류를 위해 개발자가 파라미터 조정을 해야한다.

# #3. 컴파일, 훈련
# # model.compile(loss='sparse_categorical_crossentropy',
# #               optimizer='adam',
# #               metrics=['acc'])
# # model.fit(x, y, epochs=100, validation_split=0.2)
# #다중분류에서, 원핫 하지 않았을경우 sparse 사용함. 대체로 0이라는 라벨값이 있을때만 사용한다. 원핫을 포함하고있는 loss
# model.fit(x,y)   #머신러닝은 핏에 컴파일이 포함된다.


# #4. 평가 예측
# # results = model.evaluate(x,y)
# results = model.score(x,y)   #

# print(results)  #로스와 메트릭스의 첫번째 출력
# # 머신러닝은 다 단층이다

datasets = [load_iris(return_X_y=True), load_digits(return_X_y=True), load_wine(return_X_y=True), load_breast_cancer(return_X_y=True),fetch_covtype(return_X_y=True)]
models = [RandomForestClassifier(),DecisionTreeClassifier(),LogisticRegression(),LinearSVC()]

scaler = RobustScaler()

for i, dataset in enumerate(datasets):
    x,y = dataset
    x = scaler.fit_transform(x)
    print(f"\n데이터셋 {i+1}:")
    for j, model in enumerate(models):
        model.fit(x,y)
        score = model.score(x,y)
        print(f" model {j+1}: {score:.3f}")


# model = RandomForestClassifier()
# model1 = DecisionTreeClassifier()
# model2 = LogisticRegression()
# model3 = LinearSVC()

# #LinearSVC의 파라미터 C 는 작으면 작을수록 직선을 긋는다.
#분류문제에서 딥러닝은 layer를 거쳐 선을 수정하지만, 머신러닝은 단층 레이어로 선을 그어 클래스를 분류한다.
#때문에 고도화된 문제는 완벽한 분류를 위해 개발자가 파라미터 조정을 해야한다.

# #3. 컴파일, 훈련
# # model.compile(loss='sparse_categorical_crossentropy',
# #               optimizer='adam',
# #               metrics=['acc'])
# # model.fit(x, y, epochs=100, validation_split=0.2)
# #다중분류에서, 원핫 하지 않았을경우 sparse 사용함. 대체로 0이라는 라벨값이 있을때만 사용한다. 원핫을 포함하고있는 loss
# model.fit(x,y)   #머신러닝은 핏에 컴파일이 포함된다.
# model1.fit(x,y)
# model2.fit(x,y)
# model3.fit(x,y)

# #4. 평가 예측
# # results = model.evaluate(x,y)
# results = model.score(x,y)   #
# results1 = model1.score(x,y)   #
# results2 = model2.score(x,y)   #
# results3 = model3.score(x,y)   #

# print('랜덤 포레스트 점수 : ', results)  #로스와 메트릭스의 첫번째 출력
# print('디시젼 트리 점수 : ', results1)
# print('로지스틱리그레션 점수 : ', results2)
# print('리니어 SVC 점수 : ', results3)
# 머신러닝은 다 단층이다

# Results for dataset 1:
#  model 1: 1.000
#  model 2: 1.000
#  model 3: 0.953
#  model 4: 0.947

# Results for dataset 2:
#  model 1: 1.000
#  model 2: 1.000
#  model 3: 0.997
#  model 4: 0.993

# Results for dataset 3:
#  model 1: 1.000
#  model 2: 1.000
#  model 3: 1.000
#  model 4: 1.000

# Results for dataset 4:
#  model 1: 1.000
#  model 2: 1.000
#  model 3: 0.988
#  model 4: 0.988

# Results for dataset 5:
#  model 1: 1.000
#  model 2: 1.000
#  model 3: 0.724
#  model 4: 0.713