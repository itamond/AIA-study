# 10개 데이터셋
# 10개의 파일을 만든다
#[실습/과제] 피처를 한개씩 삭제하고 성능비교
#모델은 RF로만 한다


import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,r2_score
import pandas as pd
#1. 데이터

path='./_data/ddarung/'      # .=현 폴더, study    /= 하위폴더
path_save='./_save/ddarung/'      # .=현 폴더, study    /= 하위폴더
train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0)                
test_csv = pd.read_csv(path + 'test.csv',
                       index_col=0)

train_csv = train_csv.dropna()   #dropna = 결측치 삭제 함수*****

x = train_csv.drop(['count','hour_bef_precipitation'], axis=1)    
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, random_state=337
)

#2. 모델
model = RandomForestRegressor()



#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score :', result)
y_predict = model.predict(x_test)
print('r2 :', r2_score(y_test, y_predict))

print('====================================')
print(model, ":", model.feature_importances_)

# model.score : 0.7922708636308206
# r2 : 0.7922708636308206
# ====================================
# RandomForestRegressor() : [0.58728964 0.17512251 0.01186954 0.03346998 0.03869166 0.03867172
#  0.04740425 0.04151004 0.02597066]


# model.score : 0.7732988208396263
# r2 : 0.7732988208396263
# ====================================
# RandomForestRegressor() : [0.58946529 0.17514924 0.03202073 0.04172503 0.04783001 0.04917457
#  0.03890839 0.02572673]
