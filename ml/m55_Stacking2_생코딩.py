#동일 데이터 동일 모델로 훈련하여 앙상블 시키는것이 배깅, 부스팅이다.
#앙상블 시킬때 가중치를 이전시키면 부스팅, 초기화 하면 배깅이다.
#하드voting은 '각기 다른 모델'을 훈련하고 추출된 값에 따라 단순 투표하여 결과를 뽑는 방식이다.
#소프트voting은 '각기 다른 모델'을 훈련하고 추출된 가중합끼리 비교하여 점수가 가장 높은 최종 값을 결정한다
#스태킹은 각 모델의 최종 predict 값으로 다시 훈련시키는 기법이다.
#그래서 최초 모델은 조금 약한 모델을 사용할수 있지만, 최종 predict를 뽑을때에는 고성능 모델 사용하는게 좋음
#보팅은 여러가지 모델, 배깅은 한가지 모델로 한다.


import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')



#1. 데이터
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=513,
                                                    shuffle=True,)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
# lr = LogisticRegression()
# knn = KNeighborsClassifier(n_neighbors=8)
# dt = DecisionTreeClassifier()

xg = XGBClassifier()
lg = LGBMClassifier()
cb = CatBoostClassifier(verbose=0)

models = [xg, lg, cb]
li = []
for model in models:
    model.fit(x_train,y_train)
    y_predict = model.predict(x_test)
    # print(y_predict.shape) #(114,)
    y_predict = y_predict.reshape(y_predict.shape[0], 1)
    li.append(y_predict)
    
    score = accuracy_score(y_test, y_predict)
    class_name = model.__class__.__name__
    print("{0} ACC : {1:.4f}".format(class_name,score))

# print(li) #각 모델에 대한 predict 값이 np형태의 list로 들어가있음. list형태로는 훈련 불가능
y_stacking_predict = np.concatenate(li, axis=1)
# print(aaa)
# print(aaa.shape) #(114, 3)

model = CatBoostClassifier(verbose=0)
model.fit(y_stacking_predict, y_test)
score = model.score(y_stacking_predict, y_test)
print("스태킹 결과 :", score)