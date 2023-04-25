#컬럼의 갯수가 클래스의 갯수보다 작을 때
#디폴트로 돌아가느냐


#상민이가 회귀에서 된다고 했다

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_diabetes, fetch_california_housing

from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 1. 데이터

x, y = load_diabetes(return_X_y=True)
# x, y = fetch_california_housing(return_X_y=True)
# y = np.round(y)
# print(y)
# print(np.unique(y, return_counts=True))
print(len(np.unique(y)))
# print(x.shape)  #n,32,32,3

# pca = PCA(n_components=97)
# x_train = pca.fit_transform(x_train)


lda = LinearDiscriminantAnalysis()
x_lda = lda.fit_transform(x, y)  #LDA는 y값도 필요하다.
print(x_lda.shape)

#회귀는 원래 사용할 수 없다.
#디아벳은 잘 돌아갔는데, 캘리포니아에서는 에러남.
#이유 : 회귀데이터 디아뱃의 y값은 전부 정수이다. lda가 각 정수 값을 클래스로 받아들임.
#따라서 실수형 데이터에 np.round로 정수형으로 바꿔주면 lda가 먹힌다.
#하지만 np.round는 데이터 조작
#LDA가 사용된다고 해서 사용하면 성능을 보장할 수 없다.