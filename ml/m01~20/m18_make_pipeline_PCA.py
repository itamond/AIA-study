#PCA : 주성분분석(차원축소=컬럼갯수 축소)
#컬런의 압축 개념이다. ***********삭제 개념이 아님*************
#64개의 컬런 모두를 활용하지 않는 방법
#0이라는 컬런이 많은 데이터(이미지)는 이 방법이 잘 먹히는 경우가 있다.


import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA

#1. 데이터

x, y = load_digits(return_X_y=True) #엠니스트 축소형
# print(x.shape)  #(1797, 64)   8바이 8의 숫자 이미지 데이터를 쫙 펼쳐놓은 데이터다.
# print(np.unique(y, return_counts=True))

# pca = PCA(n_components=8) #n_components = 몇개의 컬런으로 압축할 것인지
# x = pca.fit_transform(x)
# print(x.shape)


x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, random_state=337
)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델
# model = RandomForestClassifier()
# model = make_pipeline(MinMaxScaler(), RandomForestClassifier())#앞부분에 내가 사용할 스케일링, 이후에 모델 기재하면 됨
# model = make_pipeline(StandardScaler(), RandomForestClassifier())
model = make_pipeline(PCA(n_components=8),StandardScaler(), SVC())

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score :', result)
y_predict = model.predict(x_test)
print('ACC :', accuracy_score(y_test, y_predict))
