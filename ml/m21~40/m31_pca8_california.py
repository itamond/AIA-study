# PCA = 차원 축소의 개념. (컬런 압축의 개념)
# 일반적으로 x만 PCA를 적용한다.
# x만 사용하기 때문에 비지도 학습으로 분류된다. (차원축소한 결과를 y로 볼 수 있기 때문)
# 스케일링(전처리) 개념으로 볼 수도 있다.
# PCA는 각 값에 대한 선을 하나 긋고, 선쪽으로 데이터의 값을 모은다(맵핑)
# 그 다음은 선에대한 직각의 선을 긋고 같은 작업 반복


import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = fetch_california_housing()

x = datasets['data']
y = datasets.target

pca = PCA(n_components=7)   #n_components = 압축한 결과의 열 갯수

x = pca.fit_transform(x)
print(x.shape)   #넘파이는 쉐이프 표시 가능 판다스는 불가능~

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123,shuffle=True,)


# 2. 모델구성
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=123)



for i in range(5):
    pca = PCA(n_components=7-i)
    x =pca.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123,shuffle=True,)
    # 2. 모델구성
    model = RandomForestRegressor(random_state=123)

    # 3. 훈련
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print('차원', i,'개축소', '결과 :',results)
    
    
# 차원 0 개축소 결과 : 0.7786727671384369
# 차원 1 개축소 결과 : 0.7018597110810503
# 차원 2 개축소 결과 : 0.5918722922244304
# 차원 3 개축소 결과 : 0.3241551445937575
# 차원 4 개축소 결과 : 0.0789494633195541