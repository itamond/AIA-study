#linear discriminant analysis
#선형 판별 분석

#PCA는 데이터의 방향성에 따라 선을 긋는다.
#LDA는 각 데이터의 클래스별로 매치를 시킨다.
#LDA는 지도학습이다. x에 대한 각 클래스의 값을 알아야 가능한 기법.

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_digits
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 1. 데이터
# x, y = load_iris(return_X_y=True)
# x, y = load_breast_cancer(return_X_y=True)
x, y = load_digits(return_X_y=True)
# pca = PCA(n_components=3)
# x = pca.fit_transform(x)
# print(x.shape)  #(150,3)

# lda = LinearDiscriminantAnalysis()
lda = LinearDiscriminantAnalysis(n_components=2)
#n_components는 클래스의 갯수 빼기 하나 이하로 가능하다
x = lda.fit_transform(x,y)  #LDA는 y값도 필요하다.
print(x.shape, y.shape)  
#iris datasets(150, 2) (150,)  x의 컬런값이 두개로 나온 이유는 LDA가 하나의 선을 그어서 분류했기 때문이다
#LDA의 디폴트는 클래스 -1개만큼으로 압축시킨다.



import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

x=iris.data[:,2:]
y=iris.target

plt.scatter(x[:,0],x[:,1],c=y)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title('iris scatter plot')
plt.show()