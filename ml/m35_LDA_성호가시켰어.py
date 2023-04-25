#컬럼의 갯수가 클래스의 갯수보다 작을 때
#디폴트로 돌아가느냐


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_digits
from tensorflow.keras.datasets import cifar100
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 1. 데이터

(x_train,y_train),(x_test,y_test) = cifar100.load_data()
print(x_train.shape)  #n,32,32,3

x_train = x_train.reshape(-1, 32*32*3)

pca = PCA(n_components=97)
x_train = pca.fit_transform(x_train)


lda = LinearDiscriminantAnalysis(n_components=95)
# lda = LinearDiscriminantAnalysis(n_components=101)
#n_components는 클래스의 갯수 빼기 하나 이하로 가능하다
x_lda = lda.fit_transform(x_train,y_train)  #LDA는 y값도 필요하다.
print(x_lda.shape)
#iris datasets(150, 2) (150,)  x의 컬런값이 두개로 나온 이유는 LDA가 하나의 선을 그어서 분류했기 때문이다
#LDA의 디폴트는 클래스 -1개만큼으로 압축시킨다.
