#컬런의 종류에 따라 훈련 결과에 악영향을 끼치는 불필요한 컬런이 있다.
#때문에 컬런을 걸러내는 작업을 함.
#ex)PCA

#트리 계열은 결측치에 강하다.
#트리 계열은 이상치에도 강하다.
#트리 계열은 스케일링을 안해도 괜찮다.
#Nan 값이 있어도 돌아감.


import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#1. 데이터

datasets = load_iris()

x=datasets.data
y=datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, random_state=337
)

#2. 모델
# for i, value in enumerate(models):
#     model = value
#     model.fit(x_train, y_train)
#     y_predict = model.predict(x_test)
#     print('====================================')
#     print('ACC :', accuracy_score(y_test, y_predict))
#     print(model_name[i], ":", model.feature_importances_)
#     print('====================================')

model = RandomForestClassifier()

model.fit(x_train, y_train)
result = model.score(x_test,y_test)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print('XGBClassifier()', model.feature_importances_)
    

import matplotlib.pyplot as plt

def plot_feature_importances(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)
    plt.title(model)
    
plot_feature_importances(model)
plt.show()




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

