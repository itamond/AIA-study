import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from xgboost import XGBClassifier, XGBRegressor

#1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=337, train_size =0.8
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# 이상치는 결측치라고 볼 수도 있다.
# 발견된 이상치를 Nan으로 바꿔서 모두 결측치 처리해버릴 수도 잇음

# 'n_estimators' : [100, 200, 300, 400, 500, 1000] 디폴트 100 / 1~inf / 정수
# 'learning_rate' : [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001] 디폴트 0.3 / 0~1 /
# 'max_depth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10] 디폴트 6 / 0~inf/ 정수
# 'gamma' : [0, 1, 2, 3, 4, 5, 7, 10, 100] 디폴트 0 / 0~inf
# 'min_child_weight' : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100] 디폴트1 
# 'subsample' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# 'colsample_bytree' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트1 / 0~1
# 'colsample_bylevel': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트1 / 0~1
# 'colsample_bynode': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트1 / 0~1
# 'reg_alpha': [0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트0 / 0~inf / L1 절대값
# 'reg_lambda': [0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트0 / 0~inf / L2 제곱

n_splits = 5
kfold = KFold(n_splits=n_splits)
parameter ={'n_estimators' : [500],
            'learning_rate' : [0.1],
            'max_depth' : [2],
            'gamma' : [0],
            'min_child_weight' : [100],
            'subsample' : [0.7],
            'colsample_bytree' : [0],
            'colsample_bylevel': [0],
            'colsample_bynode': [0],
            'reg_alpha': [0.1],
            'reg_lambda': [0],
            }


#2. 모델

xgb = XGBRegressor(random_state=337)
model = GridSearchCV(xgb, parameter, cv=kfold, n_jobs=-1)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print("최상의 매개변수 :", model.best_params_)
print('최상의 점수 :', model.best_score_)

results = model.score(x_test, y_test)
print("최종점수 : ", results)


# 최상의 점수 : 0.4042234835070187
# 최종점수 :  0.42250819966003894