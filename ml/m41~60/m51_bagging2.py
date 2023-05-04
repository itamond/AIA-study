#동일 데이터 동일 모델로 훈련하여 앙상블 시키는것이 배깅, 부스팅이다.
#앙상블 시킬때 가중치를 이전시키면 부스팅, 초기화 하면 배깅이다.
#하드voting은 '각기 다른 모델'을 훈련하고 추출된 값에 따라 단순 투표하여 결과를 뽑는 방식이다.
#소프트voting은 '각기 다른 모델'을 훈련하고 추출된 가중합끼리 비교하여 점수가 가장 높은 최종 값을 결정한다
#스태킹은 각 모델의 최종 predict 값으로 다시 훈련시키는 기법이다.
#그래서 최초 모델은 조금 약한 모델을 사용할수 있지만, 최종 predict를 뽑을때에는 고성능 모델 사용하는게 좋음


#모델 10개 넣어서 확인해볼것


import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression  #분류다.
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier, XGBRFClassifier


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

# model =DecisionTreeClassifier()
# model = RandomForestClassifier()
# aaaa = LogisticRegression()
# aaaa = HistGradientBoostingClassifier()
aaaa = XGBRFClassifier()
model = BaggingClassifier(aaaa, # 베깅 안에 베깅 넣기 가능
                          n_estimators=10,
                          n_jobs=-1,
                          random_state=337,
                          bootstrap=True, #True가 디폴트
                          #데이터셋의 중복을 허용하는가? 중복 = 한번에 데이터 샘플에서 값을 뽑을때 중복된 값 추출을 허용하는가?
                          )
#디시젼트리를 열번 배깅한다.


#3. 훈련

model.fit(x_train,y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('acc :', model.score(x_test, y_test))
print('acc :', accuracy_score(y_test, y_pred))

# 디시젼트리
# acc : 0.9385964912280702
# acc : 0.9385964912280702

# 랜덤포레스트
# acc : 0.956140350877193
# acc : 0.956140350877193

# 베깅 디시젼트리
# acc : 0.956140350877193
# acc : 0.956140350877193

# 로지스틱 리그레션
# acc : 0.956140350877193
# acc : 0.956140350877193

# 그래디언트 부스팅 리그레션
# acc : 0.9473684210526315
# acc : 0.9473684210526315

# hist 그래디언트 부스팅 리그레션
# acc : 0.9649122807017544
# acc : 0.9649122807017544

# XGB
# acc : 0.9649122807017544
# acc : 0.9649122807017544

# XGBRF
# acc : 0.9385964912280702
# acc : 0.9385964912280702