#넘파이는 부동소수점 연산(실수)에 좋다.
#정규화 : 0부터 100만까지의 숫자를 0부터 1까지의 숫자로 바꿔줄 수 있다.  (100만으로 나누기)
#우리가 받는 데이터는 항상 x와 y값을 받고 시작한다. (그래야만 한다.)
#정규화 = nomalization (원래 단어 뜻은 다름)
#모든 x데이터를 0에서 1 사이로 만들어버린다 (y는 제외!!)
#과부화걸리지 않게함, 속도를 빠르게함, 성능도 좋아질 '수도' 있다**
#x가 나누어지더라도 가르치는 y는 동일하다. 
#(x-min)/(max-min)

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler               #MinMaxScaler=0에서 1 사이로 만든다. StandardScaler = 표준정규분포 형태로 변환시키는 정규화
#preprocessing = 전처리
from sklearn.preprocessing import MaxAbsScaler        #최대 절대값
from sklearn.preprocessing import RobustScaler        # 

#1. 데이터

datasets = load_boston()
x= datasets.data
y= datasets['target']

# print(type(x))


# print(np.min(x), np.max(x))                    #x의 최소값 보기
# scaler= MinMaxScaler()
# scaler.fit(x)                 #변환할 준비를 해라
# x = scaler.transform(x)       #변환해라
# print(np.min(x), np.max(x))

x_train, x_test, y_train, y_test = train_test_split (x,y,
                                                     train_size=0.8,
                                                     random_state=333,
                                                     )

scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)     #scaler의 fit의 범위가 x_train이라는 뜻. (x_train을 0~1로 변환)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)             # fit에서 변환한 비율에 맞춰서 x_test를 변환해라.
print(np.min(x_test), np.max(x_test))          #-0.00557837618540494 1.1478180091225068 실제로 범위 밖으로 빠진 데이터가 있다.





# 일반적으로 훈련 데이터만 정규화한다. 그 이유는 1이상, 0이하의 값을 predict 할수도 있기때문
# 전체 데이터를 정규화 하지 않는 이유는 실제로 해보면 과적합되는 경우가 많기 때문이다.
# 훈련데이터(x_train)를 정규화 한 후, 훈련데이터의 정규화 '비율'에 맞춰서 test data를 변환시킨다.
# # 또한, x_predict 또한 비율에 맞춰서 변환시킨다. x_predict값은 1을 초과할수도, 0 미만일수도 있다.
# # 때문에 train_test_split 이후에 train에만 정규화를 시행한다.

# #2. 모델
# model=Sequential()
# model.add(Dense(1, input_dim=13))


# #3. 컴파일, 훈련

# model.compile(loss = 'mse', optimizer='adam',)
# model.fit(x_train,y_train,
#           epochs=10,
#           batch_size=32,
#           verbose=1,
#           )

# #4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# print('loss :', loss)





# 각 Scaler의 차이점과 장단점은 다음과 같습니다:

# 1. StandardScaler:

# 각 feature의 평균을 0으로, 표준편차를 1로 변환합니다.
# Outlier가 존재할 경우 영향을 크게 받을 수 있습니다.
# 이상치가 없는 경우에는 대체로 잘 작동합니다.
# 선형 모델, SVM, KNN 등에서 사용됩니다.


# 2. MaxAbsScaler:

# 각 feature의 최대값의 절대값으로 나눕니다.
# 각 feature가 0과 1 사이에 위치합니다.
# Outlier에 덜 민감합니다.
# 데이터가 양수인 경우에는 대체로 잘 작동합니다.
# 선형 모델, SVM, KNN 등에서 사용됩니다.


# 3. RobustScaler:

# 중앙값(median)을 0으로, IQR(중간 50%범위)로 scale합니다.
# 이상치에 덜 민감합니다.
# 이상치가 있는 경우에 사용하는 것이 좋습니다.
# SVM 등에서 사용됩니다.


# 4. MinMaxScaler:

# 각 feature의 최소값을 0으로, 최대값을 1로 변환합니다.
# 각 feature가 0과 1 사이에 위치합니다.
# Outlier에 민감합니다.
# 데이터가 양수인 경우에는 대체로 잘 작동합니다.
# 선형 모델, SVM, KNN 등에서 사용됩니다.


# 각 Scaler는 다른 데이터 분포 및 목적에 적합합니다. 
# 따라서 적절한 Scaler를 선택해야 합니다. 예를 들어, 
# 이상치가 많은 경우 RobustScaler를 사용하는 것이 좋습니다. 
# 반면, 데이터가 음수인 경우 MaxAbsScaler 대신 StandardScaler를 사용하는 것이 좋습니다.



# 5. Normalizer()
# 추가로 Normalizer라는 방법도 알게 됐습니다. 앞의 네 가지 방법은 각 피쳐, 즉 열(columns)을 대상으로 하지만
# Normalizer의 경우 각 행(row)마다 정규화가 진행됩니다. 이는 한 행의 모든 피쳐들 사이의 유클리드 거리가
# 1이 되도록 데이터 값을 만들어줍니다. 이렇게 하면 좀 더 빠르게 학습할 수 있고, 과대적합 확률을 낮출 수 있다고
# 합니다.