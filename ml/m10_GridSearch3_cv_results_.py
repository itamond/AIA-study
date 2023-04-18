#파라미터 전체 조사
#대부분의 파라미터는 모델과 fit에서 정의함
#gridSearch는 모든 파라미터를 돌려본다
#gridSearch는 crossval 기능도 있다
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, StratifiedKFold #CV는 crossval
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import time


#1. 데이터

x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    # stratify=y,
                                                    random_state=337,
                                                    shuffle=True)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=337)
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=337)

# gamma = [0.001, 0.01, 0.1, 1, 10, 100]
# C = [0.001, 0.01, 0.1, 1, 10, 100]
# gridSearch는 딕셔너리 형태로 입력

parameters = [
    {"C":[1,10,100,1000],"kernel":['linear'],'degree':[3,4,5]},  #12번 돈다
    {'C':[1,10,100], 'kernel':['rbf','linear'], 'gamma':[0.001, 0.0001]}, #12번 돈다
    {'C':[1,10,100,1000], 'kernel':['sigmoid'], 'gamma':[0.01,0.001,0.0001], 'degree':[3,4]} #24번 돈다.
]  #총 48번 돈다, GridSearch는 단순 for문의 연속


#2. 모델
model = GridSearchCV(SVC(), parameters, 
                    #  cv=kfold, 
                     cv=5, #분류 문제에서 GridSearch의 cv 디폴트는 stratified kfold이다. 회귀문제에서는 일반 kfold
                     verbose=1,    #verbose = Fitting 5 folds for each of 48 candidates, totalling 240 fits
                     refit=True, #디폴트True. True는 최상의 파라미터로 출력. False = 최종 파라미터로 출력
                    #  refit=False, #refit은 best값을 저장시킨다. 따라서 best_관련 함수는 True에서만 사용가능
                     n_jobs=-1)#모델 정의, 파라미터 정의, cv정의
#SVC모델을 48번 돌리고, cv 5배 추가-> 따라서 240번 돔

#3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print('최적의 매개변수 :', model.best_estimator_) #가장 좋은 평가

print('최적의 파라미터 :', model.best_params_) #가장 좋은 평가

print('best_score_ :', model.best_score_) #가장 좋은 평가
#################여기까지는 트레인 데이터만 사용함################
print('model.score :', model.score(x_test,y_test))

# 최적의 매개변수 : SVC(C=1, kernel='linear')
# 최적의 파라미터 : {'C': 1, 'degree': 3, 'kernel': 'linear'}
# best_score_ : 0.9833333333333332
# model.score : 1.0
# 걸린 시간 : 2.8 초

y_predict = model.predict(x_test)
print('accuracy_score :', accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test) #model.predict와 같은 값이 나온다는건, predict에 best_estimator가 포함되어있다는 뜻
print('최적 튠 ACC:', accuracy_score(y_test, y_pred_best))
print('걸린 시간 :', np.round(end_time-start_time,2),'초')

import pandas as pd
##################################################################
# print(pd.DataFrame(model.cv_results_))  #list로 뽑으려면 DataFrame, 컬런이 하나인 상태로 출력하려면 Series로 출력
# [48 rows x 17 columns]
print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True))  
#sort_ =판다스의 정렬하는 함수. values와 index가 있음, ascending = True 는 오름차순이 디폴트. False는 내림차순
print(pd.DataFrame(model.cv_results_).columns)
# Index(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
#        'param_C', 'param_degree', 'param_kernel', 'param_gamma', 'params',
#        'split0_test_score', 'split1_test_score', 'split2_test_score',
#        'split3_test_score', 'split4_test_score', 'mean_test_score',
#        'std_test_score', 'rank_test_score'],
#       dtype='object')

path = './temp/'
pd.DataFrame(model.cv_results_)\
    .sort_values('rank_test_score', ascending=True)\
    .to_csv(path + 'm10_GridSearch3.csv')