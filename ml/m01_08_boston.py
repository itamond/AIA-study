#회귀 싹 모아서 테스트

#1. 보스턴, 캘리포니아, 따릉, 캐글바이크
import numpy as np
from sklearn.datasets import fetch_california_housing


x, y = fetch_california_housing(return_X_y=True)

print(x.shape, y.shape)   #(150, 4) (150,)


from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression       #레그레션 이름을 쓰고있지만 분류다... 시그모이드이다
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# model = RandomForestRegressor()


#LinearSVC의 파라미터 C 는 작으면 작을수록 직선을 긋는다.
#분류문제에서 딥러닝은 layer를 거쳐 선을 수정하지만, 머신러닝은 단층 레이어로 선을 그어 클래스를 분류한다.
#때문에 고도화된 문제는 완벽한 분류를 위해 개발자가 파라미터 조정을 해야한다.

#3. 컴파일, 훈련
# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer='adam',
#               metrics=['acc'])
# model.fit(x, y, epochs=100, validation_split=0.2)
#다중분류에서, 원핫 하지 않았을경우 sparse 사용함. 대체로 0이라는 라벨값이 있을때만 사용한다. 원핫을 포함하고있는 loss
# model.fit(x,y)   #머신러닝은 핏에 컴파일이 포함된다.


#4. 평가 예측
# results = model.evaluate(x,y)
# results = model.score(x,y)   #

# print(results)  #로스와 메트릭스의 첫번째 출력
# 머신러닝은 다 단층이다

models = [RandomForestRegressor(),DecisionTreeRegressor()]

for j, model in enumerate(models):
    model.fit(x,y)
    score = model.score(x,y)
    print(f" model {j+1}: {score:.3f}")
    
    
#캘리포니아
# model 1: 0.974
# model 2: 1.000

#따릉
#  model 1: 0.969
#  model 2: 1.000

#캐글바이크
#  model 1: 1.000
#  model 2: 1.000