# PCA = 차원 축소의 개념. (컬런 압축의 개념)
# 일반적으로 x만 PCA를 적용한다.
# x만 사용하기 때문에 비지도 학습으로 분류된다. (차원축소한 결과를 y로 볼 수 있기 때문)
# 스케일링(전처리) 개념으로 볼 수도 있다.



import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터
datasets = load_wine()

x = datasets['data']
y = datasets.target


print(x.shape)

for i in range(11):
    pca = PCA(n_components=11-i)
    x =pca.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123,shuffle=True,)
    # 2. 모델구성
    model = RandomForestClassifier(random_state=123)

    # 3. 훈련
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print('차원', i,'개축소', '결과 :',results)
    


# 차원 0 개축소 결과 : 0.9166666666666666
# 차원 1 개축소 결과 : 0.8888888888888888
# 차원 2 개축소 결과 : 0.9166666666666666
# 차원 3 개축소 결과 : 0.9166666666666666
# 차원 4 개축소 결과 : 0.9166666666666666
# 차원 5 개축소 결과 : 0.9166666666666666
# 차원 6 개축소 결과 : 0.9166666666666666
# 차원 7 개축소 결과 : 0.8888888888888888
# 차원 8 개축소 결과 : 0.7222222222222222
# 차원 9 개축소 결과 : 0.6388888888888888
# 차원 10 개축소 결과 : 0.6111111111111112





