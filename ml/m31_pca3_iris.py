# PCA = 차원 축소의 개념. (컬런 압축의 개념)
# 일반적으로 x만 PCA를 적용한다.
# x만 사용하기 때문에 비지도 학습으로 분류된다. (차원축소한 결과를 y로 볼 수 있기 때문)
# 스케일링(전처리) 개념으로 볼 수도 있다.



import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터
datasets = load_iris()

x = datasets['data']
y = datasets.target

print(x.shape)

for i in range(3):
    pca = PCA(n_components=x.shape[1]-i)
    x =pca.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123,shuffle=True,)
    # 2. 모델구성
    model = RandomForestClassifier(random_state=123)

    # 3. 훈련
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print('차원', i,'개축소', '결과 :',results)
    


# 차원 0 개축소 결과 : 0.9333333333333333
# 차원 1 개축소 결과 : 0.9333333333333333
# 차원 2 개축소 결과 : 0.8333333333333334



