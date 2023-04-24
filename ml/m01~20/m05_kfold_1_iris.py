#cross_val = 교차검증의 사용법에 관하여

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
#1. 데이터
x, y = load_iris(return_X_y=True)

# x_train, x_test, y_train, y_test = train_test_split(x, y,
#                                                     shuffle=True,
#                                                     random_state=123,
#                                                     test_size=0.2)
#import 한것을 정의

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle=True, random_state=123)  #n_splits 나눌 갯수 shuffle 섞어서 나눈다.
#kfold = KFold() 디폴트가 5. 데이터를 훈련시키는 위치에 따라서 성능차이 엄청남
#기존에는 test 데이터를, train데이터와 다른 데이터로 구분함으로써 평가 용도로 사용하였는데 이를 구분없이 사용하면 이 또한 과적합이라 볼 수 있다.
#따라서 테스트 데이터를 분리한 후, cross_val 시켜주는 방법도 있다.


#2. 모델
model = ExtraTreesClassifier()

#3, 4 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold)  #cross_val의 스코어를 빼준다. (모델, 데이터, 크로스발리데이션을 어떤것을 쓸 것인지cv=kfold)
# scores = cross_val_score(model, x, y, cv=5) 가능. 

#kfold의 갯수만큼 훈련을 시킨다.
#[0.96666667 1.         0.93333333 0.93333333 0.9       ]

print('ACC :', scores, '\ncross_val_score 평균 : ', round(np.mean(scores), 4))
# ACC : [0.96666667 1.         0.93333333 0.93333333 0.9       ]
#  cross_val_score 평균 :  0.9467





# 데이터셋 1: load_iris
# ACC : [0.93333333 0.96666667 0.93333333 0.93333333 0.93333333] 
# cross_val_score 평균 :  0.94

# 데이터셋 2: load_digits
# ACC : [0.97777778 0.975      0.97493036 0.98050139 0.98607242] 
# cross_val_score 평균 :  0.9789

# 데이터셋 3: load_wine
# ACC : [1.         1.         0.97222222 1.         0.94285714] 
# cross_val_score 평균 :  0.983

# 데이터셋 4: load_breast_cancer
# ACC : [0.99122807 0.97368421 0.93859649 0.96491228 0.92920354] 
# cross_val_score 평균 :  0.9595

# 데이터셋 5: fetch_covtype
# ACC : [0.95393406 0.95454506 0.95566341 0.95619697 0.9559474 ] 
# cross_val_score 평균 :  0.9553
