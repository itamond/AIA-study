#실습

#데이터

#분류 아이리스 캔서 데이콘디아벳 와인 펫치코브타입 디짓스
#회귀 디아벳스 캘리포니아 데이콘따릉 캐글바이크

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

#트레인 테스트 분리, 스케일러 적용




import numpy as np
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, KFold, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import time
import pandas as pd


path='./_data/ddarung/'      # .=현 폴더, study    /= 하위폴더
path_save='./_save/ddarung/'      # .=현 폴더, study    /= 하위폴더
train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0)                
test_csv = pd.read_csv(path + 'test.csv',
                       index_col=0)

train_csv = train_csv.dropna()   #dropna = 결측치 삭제 함수*****

x = train_csv.drop(['count'], axis=1)    
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=337,
                                                    shuffle=True)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=337)

model = HalvingGridSearchCV(RandomForestRegressor(), parameters,
                     cv=kfold, 
                     verbose=1, 
                     refit=True, 
                     n_jobs=-1)

stt = time.time()

model.fit(x_train, y_train)

ett = time.time()



print('최적의 매개변수 :', model.best_estimator_) #가장 좋은 평가

print('최적의 파라미터 :', model.best_params_) #가장 좋은 평가

print('best_score_ :', model.best_score_) #가장 좋은 평가

print('model.score :', model.score(x_test,y_test))

y_predict = model.predict(x_test)

y_pred_best = model.best_estimator_.predict(x_test) 

print('r2_score :', r2_score(y_test, y_predict))


print('최적 튠 r2:', r2_score(y_test, y_pred_best))

print('걸린 시간 :', round(ett-stt,2),'초')


# GridSearch 
# 최적의 매개변수 : RandomForestRegressor(max_depth=8, min_samples_leaf=10, min_samples_split=10)
# 최적의 파라미터 : {'max_depth': 8, 'min_samples_leaf': 10, 'min_samples_split': 10, 'n_estimators': 100}
# best_score_ : 0.44900864341774127
# model.score : 0.4809467451965985
# r2_score : 0.4809467451965985
# 최적 튠 r2: 0.4809467451965985
# 걸린 시간 : 14.0 초


# RandomSearch
# 최적의 매개변수 : RandomForestRegressor(max_depth=12, min_samples_leaf=3, min_samples_split=3,
#                       n_estimators=200)
# 최적의 파라미터 : {'n_estimators': 200, 'min_samples_split': 3, 'min_samples_leaf': 3, 'max_depth': 12}
# best_score_ : 0.7401079097340809
# model.score : 0.7806187172992822
# r2_score : 0.7806187172992822
# 최적 튠 r2: 0.7806187172992822
# 걸린 시간 : 8.3 초


# HalvingGridSearch
# 최적의 매개변수 : RandomForestRegressor(max_depth=12, min_samples_leaf=3, min_samples_split=3,
#                       n_estimators=200)
# 최적의 파라미터 : {'max_depth': 12, 'min_samples_leaf': 3, 'min_samples_split': 3, 'n_estimators': 200}
# best_score_ : 0.7390494291872916
# model.score : 0.7879346222255889
# r2_score : 0.7879346222255889
# 최적 튠 r2: 0.7879346222255889
# 걸린 시간 : 19.67 초