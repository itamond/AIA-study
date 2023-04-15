import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, fetch_covtype



#1. 데이터
# datasets = load_iris()
# x = datasets.data
# y = datasets['target']
x, y = fetch_covtype(return_X_y=True)

print(x.shape, y.shape)   #(150, 4) (150,)

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
# model.add(Dense(3, activation='softmax'))
# model = LinearSVC()  #알고리즘 연산이 다 포함되어있다. 나중에 다 공부해라. 어떤 모델인지 정도는 알아야한다.
model = RandomForestClassifier()
model1 = DecisionTreeClassifier()
model2 = LogisticRegression()
model3 = LinearSVC()

#LinearSVC의 파라미터 C 는 작으면 작을수록 직선을 긋는다.
#분류문제에서 딥러닝은 layer를 거쳐 선을 수정하지만, 머신러닝은 단층 레이어로 선을 그어 클래스를 분류한다.
#때문에 고도화된 문제는 완벽한 분류를 위해 개발자가 파라미터 조정을 해야한다.

#3. 컴파일, 훈련
# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer='adam',
#               metrics=['acc'])
# model.fit(x, y, epochs=100, validation_split=0.2)
#다중분류에서, 원핫 하지 않았을경우 sparse 사용함. 대체로 0이라는 라벨값이 있을때만 사용한다. 원핫을 포함하고있는 loss
model.fit(x,y)   #머신러닝은 핏에 컴파일이 포함된다.
model1.fit(x,y)
model2.fit(x,y)
model3.fit(x,y)

#4. 평가 예측
# results = model.evaluate(x,y)
results = model.score(x,y)   #
results1 = model1.score(x,y)   #
results2 = model2.score(x,y)   #
results3 = model3.score(x,y)   #

print('랜덤 포레스트 점수 : ', results)  #로스와 메트릭스의 첫번째 출력
print('디시젼 트리 점수 : ', results1)
print('로지스틱리그레션 점수 : ', results2)
print('리니어 SVC 점수 : ', results3)
# 머신러닝은 다 단층이다


# 2.breast_cancer
# 랜덤 포레스트 점수 :  1.0
# 디시젼 트리 점수 :  1.0
# 로지스틱리그레션 점수 :  0.9472759226713533
# 리니어 SVC 점수 :  0.9015817223198594

# 3.wine
# 랜덤 포레스트 점수 :  1.0
# 디시젼 트리 점수 :  1.0
# 로지스틱리그레션 점수 :  0.9662921348314607
# 리니어 SVC 점수 :  0.9550561797752809

# 4.digits
# 랜덤 포레스트 점수 :  1.0
# 디시젼 트리 점수 :  1.0
# 로지스틱리그레션 점수 :  1.0
# 리니어 SVC 점수 :  0.9816360601001669

# 5.fetch_covtype

# 6. dacon_diabetes