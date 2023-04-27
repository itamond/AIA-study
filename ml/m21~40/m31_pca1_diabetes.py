# PCA = 주성분 분석, 차원 축소의 개념. (컬런 압축의 개념)
# 일반적으로 x만 PCA를 적용한다.
# x만 사용하기 때문에 비지도 학습으로 분류된다. (차원축소한 결과를 y로 볼 수 있기 때문)
# 스케일링(전처리) 개념으로 볼 수도 있다.
# PCA는 각 값에 대한 선을 하나 긋고, 선쪽으로 데이터의 값을 모은다(맵핑)
# 그 다음은 선에대한 직각의 선을 긋고 같은 작업 반복
# 

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = load_diabetes()

x = datasets['data']
y = datasets.target

print(x.shape)

# pca = PCA(n_components=7)   #n_components = 압축한 결과의 열 갯수

# x = pca.fit_transform(x)
# print(x.shape)   #넘파이는 쉐이프 표시 가능 판다스는 불가능~

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123,shuffle=True,)


# 2. 모델구성
from sklearn.ensemble import RandomForestRegressor

for i in range(9):
    pca = PCA(n_components=10-i)
    x =pca.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123,shuffle=True,)
    # 2. 모델구성
    model = RandomForestRegressor(random_state=123)

    # 3. 훈련
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print('차원', i,'개축소', '결과 :',results)


# 차원 0 개축소 결과 : 0.5183107084523279
# 차원 1 개축소 결과 : 0.5180342280158161
# 차원 2 개축소 결과 : 0.5123839178379545
# 차원 3 개축소 결과 : 0.5141328515687419
# 차원 4 개축소 결과 : 0.47972029709825104
# 차원 5 개축소 결과 : 0.4973618124562733
# 차원 6 개축소 결과 : 0.4632452843627035
# 차원 7 개축소 결과 : 0.20432135958216757
# 차원 8 개축소 결과 : 0.0870385988925938