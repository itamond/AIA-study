#파라미터 전체 조사
#대부분의 파라미터는 모델과 fit에서 정의함
#gridSearch는 모든 파라미터를 돌려본다
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
#1. 데이터

x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    stratify=y,
                                                    random_state=337,
                                                    shuffle=True)

gamma = [0.001, 0.01, 0.1, 1, 10, 100]
C = [0.001, 0.01, 0.1, 1, 10, 100]

max_score=0

for i in gamma:
    for j in C:
        #2. 모델
        model = SVC(gamma=i, C=j)
        
        #3. 컴파일, 훈련
        model.fit(x_train, y_train)
        
        #4. 평가, 예측
        #텐서플로의 evaluate
        #사이킷런의 score
        score = model.score(x_test, y_test)
                
        if max_score < score :
            max_score = score
            best_parameters = {'gamma' : i, 'C' : j} # if문은 if문끼리 묶여있다. 점수가 갱신될때만 best_parameters도 갱신된다.


print('최고점수 :', max_score)
print('최적의 매개변수 :', best_parameters) #매개변수=파라미터


# 최고점수 : 1.0
# 최적의 매개변수 : {'gamma': 10, 'C': 1}


