#m07파일 복붙
#문제점 1. train, test/ test로 predict 한 것이므로 과적합만큼 결과의 acc가 안나올 수 있음
#문제점 2. stratify kfold 과정중에 y값이 편향될 수 있다.
#StratifiedKFold = 비율대로 잘라주는 kfold. 위 문제점을 해결할 수 있다.
#분류 문제에서만 사용함.
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=337,  #stratify=y,
)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=337,)

#2. 모델 구성
model = SVC()

#3, 4 컴파일, 훈련, 평가, 예측

score = cross_val_score(model, x_train, y_train, cv = kfold)
print('cross_val_score :', score,
      '\n교차검증평균점수 :', round(np.mean(score),4))

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print('cross_val_predict ACC : ', accuracy_score(y_test, y_predict))

# cross_val_score : [1.         0.95652174 1.         0.90909091 0.95454545] 
# 교차검증평균점수 : 0.964
# cross_val_predict ACC :  0.9736842105263158


