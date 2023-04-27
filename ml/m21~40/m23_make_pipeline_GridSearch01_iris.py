#랜덤서치, 그리드서치, 할빙그리드서치를
#for문으로 한방에 넣어라
#단, 패치코브타입처럼 느린놈은 랜덤이나 할빙중에 하나만 넣어라

#n_iter 5 cv 2


import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits, fetch_covtype, load_diabetes, fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV

#1. 데이터

data_list = [load_iris,
             load_breast_cancer, 
             load_wine, 
             load_digits]

data_name_list = ['아이리스',
                  '브레스트 캔서',
                  '와인',
                  '디짓스']

Grid_list = [GridSearchCV,
             RandomizedSearchCV,
             HalvingGridSearchCV,
             HalvingRandomSearchCV]

Grid_name = ['그리드서치',
             '랜더마이즈서치',
             '할빙그리드서치',
             '할빙랜덤서치']

scaler_list = [MinMaxScaler(),
               RobustScaler(),
               StandardScaler(),
               MaxAbsScaler()]

scaler_name = ['민맥스',
               '로버스트',
               '스탠다드',
               '맥스앱스']

# x_train, x_test, y_train, y_test = train_test_split(
#     x,y, train_size=0.8, shuffle=True, random_state=337
# )

parameters = [
    {'randomforestclassifier__n_estimators':[100, 200]}, 
    {'randomforestclassifier__max_depth':[6, 8, 10, 12]},
    {'randomforestclassifier__min_samples_leaf':[3, 5, 7, 10]},
    {'randomforestclassifier__min_samples_split':[2, 3, 5, 10]},
]
parameters =[
    {'randomforestclassifier__n_estimators':[100],'randomforestclassifier__max_depth':[6,8,10,12],'randomforestclassifier__min_samples_leaf':[3,10],'randomforestclassifier__min_samples_split':[2,10]},
    {'randomforestclassifier__n_estimators':[100],'randomforestclassifier__max_depth':[6,8,10,12],'randomforestclassifier__min_samples_leaf':[5,7],'randomforestclassifier__min_samples_split':[3,5]},
    {'randomforestclassifier__n_estimators':[200],'randomforestclassifier__max_depth':[6,8],'randomforestclassifier__min_samples_leaf':[7,10],'randomforestclassifier__min_samples_split':[5,10]},
    {'randomforestclassifier__n_estimators':[200],'randomforestclassifier__max_depth':[10,12],'randomforestclassifier__min_samples_leaf':[3,5],'randomforestclassifier__min_samples_split':[2,3,]}    
]



#2. 모델
pipe = make_pipeline(StandardScaler(), RandomForestClassifier())


# model = GridSearchCV(pipe, parameters,
#                      cv = 5,
#                      verbose=1,
#                      n_jobs=-1
#                      )

for i in range(len(data_list)):
    x, y = data_list[i](return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state= 337, stratify=y)
    max_score = 0
    
    for j, value2 in enumerate(scaler_list):
        
        for k, value3 in enumerate(Grid_list):                        
            model = value3(pipe, parameters,
                     cv = 5,
                     verbose=1,
                     n_jobs=-1
                     )
            model.fit(x_train, y_train)
            score = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            acc=accuracy_score(y_test, y_pred)
            
            if max_score < score:
                max_score = score
                max_s_name = scaler_name[j]
                max_model_name = Grid_name[k]
    
    print('\n')
    print('===============',data_name_list[i],'================')
    print('최고모델 :', max_s_name, max_model_name, max_score)
    print('=============================================')



# =============== 아이리스 ================
# 최고모델 : 민맥스 그리드서치 0.9666666666666667
# =============================================



# =============== 브레스트 캔서 ================
# 최고모델 : 민맥스 그리드서치 0.9473684210526315
# =============================================



# =============== 와인 ================
# 최고모델 : 민맥스 랜더마이즈서치 1.0
# =============================================


# =============== 디짓스 ================
# 최고모델 : 로버스트 할빙그리드서치 0.975
# =============================================
