# 오늘 배운 결측치처리를 마음껏 활용하여
# 성능 향상


# PCA = 차원 축소의 개념. (컬런 압축의 개념)
# 일반적으로 x만 PCA를 적용한다.
# x만 사용하기 때문에 비지도 학습으로 분류된다. (차원축소한 결과를 y로 볼 수 있기 때문)
# 스케일링(전처리) 개념으로 볼 수도 있다.
# PCA는 각 값에 대한 선을 하나 긋고, 선쪽으로 데이터의 값을 모은다(맵핑)
# 그 다음은 선에대한 직각의 선을 긋고 같은 작업 반복


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.experimental import enable_iterative_imputer 
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer #결측치에 대한 책임을 돌린다
from sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터

path='./_data/ddarung/'      # .=현 폴더, study    /= 하위폴더
path_save='./_save/ddarung/'      # .=현 폴더, study    /= 하위폴더
train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0)                
test_csv = pd.read_csv(path + 'test.csv',
                       index_col=0)

# train_csv = train_csv.dropna()   #dropna = 결측치 삭제 함수*****

x = train_csv.drop(['count','hour_bef_precipitation'], axis=1)    
test_csv = test_csv.drop(['hour_bef_precipitation'], axis=1)
y = train_csv['count']

imputer = IterativeImputer(estimator=XGBRegressor())
# imputer = IterativeImputer(estimator=DecisionTreeRegressor())
x = imputer.fit_transform(x)
test_csv = imputer.transform(test_csv)


# pca = PCA(n_components=7)   #n_components = 압축한 결과의 열 갯수

# x = pca.fit_transform(x)
# print(x.shape)   #넘파이는 쉐이프 표시 가능 판다스는 불가능~

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123,shuffle=True,)


# 2. 모델구성
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=123)



for i in range(5):
    pca = PCA(n_components=8-i)
    x =pca.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123,shuffle=True,)
    # 2. 모델구성
    model = RandomForestRegressor(random_state=123)

    # 3. 훈련
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print('차원', i,'개축소', '결과 :',results)



# 차원 0 개축소 결과 : 0.6910154148035434
# 차원 1 개축소 결과 : 0.6751331786552022
# 차원 2 개축소 결과 : 0.6931804470730041
# 차원 3 개축소 결과 : 0.6221375725505367
# 차원 4 개축소 결과 : 0.2949010441540544


# 임퓨터 적용 후 각 PCA 결과
# 차원 0 개축소 결과 : 0.6797414055645661
# 차원 1 개축소 결과 : 0.6854656547003412
# 차원 2 개축소 결과 : 0.7137086450070921******** 최고값
# 차원 3 개축소 결과 : 0.628704286893147
# 차원 4 개축소 결과 : 0.3510701218622868