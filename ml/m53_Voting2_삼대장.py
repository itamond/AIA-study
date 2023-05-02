#동일 데이터 동일 모델로 훈련하여 앙상블 시키는것이 배깅, 부스팅이다.
#앙상블 시킬때 가중치를 이전시키면 부스팅, 초기화 하면 배깅이다.
#하드voting은 '각기 다른 모델'을 훈련하고 추출된 값에 따라 단순 투표하여 결과를 뽑는 방식이다.
#소프트voting은 '각기 다른 모델'을 훈련하고 추출된 가중합끼리 비교하여 점수가 가장 높은 최종 값을 결정한다
#스태킹은 각 모델의 최종 predict 값으로 다시 훈련시키는 기법이다.
#그래서 최초 모델은 조금 약한 모델을 사용할수 있지만, 최종 predict를 뽑을때에는 고성능 모델 사용하는게 좋음

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')
#1. 데이터
x, y = fetch_california_housing(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=513,
                                                    shuffle=True,)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
xg = XGBRegressor()
lg = LGBMRegressor()
cat = CatBoostRegressor(verbose=0)


model = VotingRegressor(
    estimators=[('XGB', xg),('LG', lg),('CAT', cat)],
    verbose=0 # Regressor에서 voting이 안먹힌다
) #보팅 디폴트 하드. 소프트가 성능이 더 좋다고 한다

#3. 훈련

model.fit(x_train,y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('r2 :', model.score(x_test, y_test))
print('r2 :', r2_score(y_test, y_pred))

# 디시젼트리
# acc : 0.9385964912280702
# acc : 0.9385964912280702

# 랜덤포레스트
# acc : 0.956140350877193
# acc : 0.956140350877193

# 베깅 디시젼트리
# acc : 0.956140350877193
# acc : 0.956140350877193


# 하드보팅
# acc : 0.9649122807017544
# acc : 0.9649122807017544

# 소프트보팅
# acc : 0.9649122807017544
# acc : 0.9649122807017544

regressor = [xg, lg, cat]
for model2 in regressor :
    model2.fit(x_train,y_train)
    y_predict = model2.predict(x_test)
    score2 = r2_score(y_test, y_predict)
    class_name = model2.__class__.__name__  # 
    print("{0} R2 :{1:.4f}".format(class_name, score2))#{0}정확도 :{1:.4f}를 출력하겠다. 중괄호 안에 변수 가능. 뒤에 지정한 class_name, score2가 들어감


    
# print(model.__class__.__name__)# 이름만 뽑는법