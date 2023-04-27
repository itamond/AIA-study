#컬런의 종류에 따라 훈련 결과에 악영향을 끼치는 불필요한 컬런이 있다.
#때문에 컬런을 걸러내는 작업을 함.
#ex)PCA

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#1. 데이터

x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, random_state=337
)

#2. 모델
models = [DecisionTreeClassifier(),RandomForestClassifier(),GradientBoostingClassifier(),XGBClassifier()]
model_name = ['디시젼트리',
              '랜덤포레스트',
              '그레디언트부스팅',
              'XGB클래지파이어']


# for i, value in enumerate(models):
#     model = value
#     model.fit(x_train, y_train)
#     y_predict = model.predict(x_test)
#     print('====================================')
#     print('ACC :', accuracy_score(y_test, y_predict))
#     print(model_name[i], ":", model.feature_importances_)
#     print('====================================')
    
    
    


for i,v in enumerate(models):
    model = v
    model.fit(x_train, y_train)
    result = model.score(x_test,y_test)
    y_predict = model.predict(x_test)
    acc = accuracy_score(y_test, y_predict)
    print(model_name[i], ':', "acc", acc)
    if i !=3:
        print(model, ':', '컬럼별 중요도',model.feature_importances_)
    else :
        print('XGBClassifier()', model.feature_importances_)
    print('----------------------------------------------')
    

# ====================================
# ACC : 0.9333333333333333
# 디시젼트리 : [0.01671193 0.03342386 0.38987262 0.5599916 ]
# ====================================
# ====================================
# ACC : 0.9666666666666667
# 랜덤포레스트 : [0.12053831 0.0353449  0.46390512 0.38021167]
# ====================================
# ====================================
# ACC : 0.9666666666666667
# 그레디언트부스팅 : [0.00565822 0.01396963 0.80318078 0.17719137]
# ====================================
# ====================================
# ACC : 0.9666666666666667
# XGB클래지파이어 : [0.01794496 0.01218657 0.8486943  0.12117416]
# ====================================

