# PCA = 차원 축소의 개념. (컬런 압축의 개념)
# 일반적으로 x만 PCA를 적용한다.
# x만 사용하기 때문에 비지도 학습으로 분류된다. (차원축소한 결과를 y로 볼 수 있기 때문)
# 스케일링(전처리) 개념으로 볼 수도 있다.



import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터
datasets = load_breast_cancer()

x = datasets['data']
y = datasets.target

# print(x.shape)

pca = PCA(n_components=30)
x = pca.fit_transform(x)
# print(x.shape)

pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)


# print(np.cumsum(pca_EVR))
pca_cumsum = np.cumsum(pca_EVR)
import matplotlib.pyplot as plt
plt.plot(pca_cumsum)
plt.grid()
plt.show()


# cumsum = 누적합
# [9.82044672e-01 1.61764899e-02 1.55751075e-03 1.20931964e-04
#  8.82724536e-05 6.64883951e-06 4.01713682e-06 8.22017197e-07
#  3.44135279e-07 1.86018721e-07 6.99473205e-08 1.65908880e-08
#  6.99641650e-09 4.78318306e-09 2.93549214e-09 1.41684927e-09
#  8.29577731e-10 5.20405883e-10 4.08463983e-10 3.63313378e-10
#  1.72849737e-10 1.27487508e-10 7.72682973e-11 6.28357718e-11
#  3.57302295e-11 2.76396041e-11 8.14452259e-12 6.30211541e-12
#  4.43666945e-12 1.55344680e-12]
# 값의 합은 0.9999999999999998

# cumsum
# [0.98204467 0.99822116 0.99977867 0.9998996  0.99998788 0.99999453
#  0.99999854 0.99999936 0.99999971 0.99999989 0.99999996 0.99999998
#  0.99999999 0.99999999 1.         1.         1.         1.
#  1.         1.         1.         1.         1.         1.
#  1.         1.         1.         1.         1.         1.        ]
# 각 PCA의 진행에 따른 원본과의 일치율
# 15번째부터 1의 값임.
# 15번째 pca부터 원본과 동일한 값이 나온다는 뜻
# 15번째까지는 데이터의 축소에 따른 손실이 없다고 판단