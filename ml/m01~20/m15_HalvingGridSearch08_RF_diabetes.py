#파라미터 전체 조사
#대부분의 파라미터는 모델과 fit에서 정의함
#gridSearch는 모든 파라미터를 돌려본다
#gridSearch는 crossval 기능도 있다
#RandomSearch는 fold 하나당 x개씩만 랜덤하게 뽑아서 훈련
#HalvingGridSearch    GridSearch가 모든 데이터를 계산하여 시간이 느린 반면

import numpy as np
from sklearn.datasets import load_iris, load_digits, load_diabetes
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, StratifiedKFold, train_test_split #CV는 crossval
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import time
from sklearn.ensemble import RandomForestRegressor

import pandas as pd


#1. 데이터

x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    # stratify=y,
                                                    random_state=1234,
                                                    shuffle=True)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=337)
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=337)

# gamma = [0.001, 0.01, 0.1, 1, 10, 100]
# C = [0.001, 0.01, 0.1, 1, 10, 100]
# gridSearch는 딕셔너리 형태로 입력


parameters = [
    {'n_estimators':[100, 200]},  #에포와 동일한 기능
    {'max_depth':[6, 8, 10, 12]},
    {'min_samples_leaf':[3, 5, 7, 10]},
    {'min_samples_split':[2, 3, 5, 10]},
    {'n_jobs':[-1, 2, 4]}    
]
parameters =[
    {'n_estimators':[100],'max_depth':[6,8,10,12],'min_samples_leaf':[3,10],'min_samples_split':[2,10]},
    {'n_estimators':[100],'max_depth':[6,8,10,12],'min_samples_leaf':[5,7],'min_samples_split':[3,5]},
    {'n_estimators':[200],'max_depth':[6,8],'min_samples_leaf':[7,10],'min_samples_split':[5,10]},
    {'n_estimators':[200],'max_depth':[10,12],'min_samples_leaf':[3,5],'min_samples_split':[2,3,]}    
]


#2. 모델
# model = GridSearchCV(SVC(), parameters,  #그리드서치에서 랜덤하게 뽑아서 쓴다.
model = HalvingGridSearchCV(RandomForestRegressor(), parameters,  #그리드서치에서 랜덤하게 뽑아서 쓴다.
                    #  cv=kfold, 
                     cv=4, #분류 문제에서 GridSearch의 cv 디폴트는 stratified kfold이다. 회귀문제에서는 일반 kfold
                     verbose=1,    #verbose = Fitting 5 folds for each of 48 candidates, totalling 240 fits
                     refit=True, #디폴트True. True는 최상의 파라미터로 출력. False = 최종 파라미터로 출력
                    #  refit=False, #refit은 best값을 저장시킨다. 따라서 best_관련 함수는 True에서만 사용가능
                     n_jobs=-1,
                     factor=3.5,       #2.4와 같은 소수점 입력도 들어간다.
                     )#모델 정의, 파라미터 정의, cv정의
#SVC모델을 48번 돌리고, cv 5배 추가-> 따라서 240번 돔

#3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

# print('걸린 시간 :', np.round(end_time-start_time,2),'초')

# print(x.shape, x_train.shape) # (1797, 64) (1437, 64)


# n_iterations: 3
# n_required_iterations: 4
# n_possible_iterations: 3      1437개의 데이터를, 100개로 시작해서 factor만큼 데이터가 증폭되면 3번의 훈련만 가능.
# min_resources_: 80           #모델이 임의로 정한 갯수
# max_resources_: 1437          #트레인 데이터의 갯수
# aggressive_elimination: False
# factor: 3                  #요인  iter가 지날때마다  candidates가 factor만큼 나눠진다. 또 resources는 factor만큼 배가 된다. #디폴트 3
# ----------
# iter: 0                    # x번째 반복
# n_candidates: 48
# n_resources: 80
# Fitting 4 folds for each of 48 candidates, totalling 192 fits       #연산 후 좋은 결과값을 갖고 다음 iter로 넘어감.
# ----------
# iter: 1                    # x번째 반복
# n_candidates: 16
# n_resources: 240
# Fitting 4 folds for each of 16 candidates, totalling 64 fits
# ----------
# iter: 2                    # x번째 반복
# n_candidates: 6
# n_resources: 720
# Fitting 4 folds for each of 6 candidates, totalling 24 fits
# 걸린 시간 : 3.05 초
# (1797, 64) (1437, 64)





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
from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
print('r2 :', r2_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test) #model.predict와 같은 값이 나온다는건, predict에 best_estimator가 포함되어있다는 뜻
print('최적 튠 r2:', r2_score(y_test, y_pred_best))
print('걸린 시간 :', np.round(end_time-start_time,2),'초')




##################################################################
# print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True))  
# print(pd.DataFrame(model.cv_results_).columns)

# path = './temp/'
# pd.DataFrame(model.cv_results_)\
#     .sort_values('rank_test_score', ascending=True)\
#     .to_csv(path + 'm14_HalvingGridSearch1.csv')


# GridSearch
# 최적의 매개변수 : RandomForestRegressor(max_depth=8, min_samples_leaf=10, min_samples_split=10)
# 최적의 파라미터 : {'max_depth': 8, 'min_samples_leaf': 10, 'min_samples_split': 10, 'n_estimators': 100}
# best_score_ : 0.44900864341774127
# model.score : 0.4809467451965985
# r2_score : 0.4809467451965985
# 최적 튠 r2: 0.4809467451965985
# 걸린 시간 : 14.0 초

# RandomSearch
# 최적의 매개변수 : RandomForestRegressor(max_depth=6, min_samples_leaf=10, min_samples_split=10)
# 최적의 파라미터 : {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 10, 'max_depth': 6}
# best_score_ : 0.4589966384948346
# model.score : 0.45676358251464233
# r2_score : 0.45676358251464233
# 최적 튠 r2: 0.45676358251464233
# 걸린 시간 : 6.26 초


# HalvingGridSearch
# 최적의 매개변수 : RandomForestRegressor(max_depth=12, min_samples_leaf=5, min_samples_split=3,
#                       n_estimators=200)
# 최적의 파라미터 : {'max_depth': 12, 'min_samples_leaf': 5, 'min_samples_split': 3, 'n_estimators': 200}
# best_score_ : 0.41664051737542007
# model.score : 0.4482178064725566
# r2 : 0.4482178064725566
# 최적 튠 r2: 0.4482178064725566
# 걸린 시간 : 12.99 초