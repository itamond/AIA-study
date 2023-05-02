# [실습] 각종 모델 넣기
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import all_estimators
from sklearn.ensemble import BaggingClassifier, VotingClassifier, VotingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
data_list = [load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes]


path='./_data/ddarung/'      # .=현 폴더, study    /= 하위폴더
path_save='./_save/ddarung/'      # .=현 폴더, study    /= 하위폴더
train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0)                
test_csv = pd.read_csv(path + 'test.csv',
                       index_col=0)

train_csv = train_csv.dropna()   #dropna = 결측치 삭제 함수*****

x = train_csv.drop(['count'], axis=1)    
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123, train_size=0.8, shuffle=True)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# 2. 모델
xg = XGBRegressor()
lg = LGBMRegressor()
cat = CatBoostRegressor(verbose=0)


model = VotingRegressor(
    estimators=[('XGB', xg),('LG', lg),('CAT', cat)],
    verbose=0 # Regressor에서 voting이 안먹힌다
) #보팅 디폴트 하드. 소프트가 성능이 더 좋다고 한다


model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# 4. 평가, 예측
regressor = [xg, lg, cat]
r2 = r2_score(y_test, y_pred)
print('r2 :', r2)

for model2 in regressor :
    model2.fit(x_train,y_train)
    y_predict = model2.predict(x_test)
    score2 = r2_score(y_test, y_predict)
    class_name = model2.__class__.__name__  # 
    print("{0} R2 :{1:.4f}".format(class_name, score2))#{0}정확도 :{1:.4f}를 출력하겠다. 중괄호 안에 변수 가능. 뒤에 지정한 class_name, score2가 들어감

# r2 : 0.8073618898271593
# XGBRegressor R2 :0.7752
# LGBMRegressor R2 :0.7912
# CatBoostRegressor R2 :0.8180
