# 삼중포문!!
# 앞부분에는 데이터셋
# 두번째는 스케일러
# 세번째는 모델
# 스케일러 전부
# 모델 = 랜덤, SVC, 디시젼트리

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits, fetch_covtype, load_diabetes, fetch_california_housing
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
warnings.filterwarnings(action='ignore')
from sklearn.metrics import r2_score, accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline



data_list = [load_diabetes,
             fetch_california_housing             
             ]

data_name_list = ['디아벳',
                  '캘리포니아_하우징',
                  ]

scaler_list = [MinMaxScaler(),
               RobustScaler(),
               StandardScaler(),
               MaxAbsScaler()]

scaler_name = ['민맥스',
               '로버스트',
               '스탠다드',
               '맥스앱스']

model_list = [RandomForestRegressor(),
              DecisionTreeRegressor(),
              ]

model_name_list = ['RandomForestRegressor :',
                   'DecisionTreeRegressor :']

max_data_name = '바보'
max_scaler_name = '바보'
max_model_name = '바보'
for i in range(len(data_list)):
    x, y = data_list[i](return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state= 337)
    max_score = 0
    for j, value2 in enumerate(scaler_list):
        
        for k, value3 in enumerate(model_list):
            
            model = Pipeline([('scaler',value2), ('model',value3)])
            model.fit(x_train, y_train)
            score = model.score(x_test, y_test)
            y_pred=model.predict(x_test)
            r2=r2_score(y_test, y_pred)
            
            if max_score < score:
                max_score = score
                max_s_name = scaler_name[j]
                max_model_name = model_name_list[k]
    
    print('\n')
    print('===============',data_name_list[i],'================')
    print('최고모델 :', max_s_name, max_model_name, max_score)
    print('=============================================')



# makepipeline
# =============== 디아벳 ================
# 최고모델 : 맥스앱스 RandomForestRegressor : 0.4170849446015845
# =============================================


# =============== 캘리포니아_하우징 ================
# 최고모델 : 맥스앱스 RandomForestRegressor : 0.8020198378844008
# =============================================


# Pipeline
# =============== 디아벳 ================
# 최고모델 : 민맥스 RandomForestRegressor : 0.4189783921983017
# =============================================


# =============== 캘리포니아_하우징 ================
# 최고모델 : 맥스앱스 RandomForestRegressor : 0.799929907402232
# =============================================