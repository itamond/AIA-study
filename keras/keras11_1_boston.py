
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# print(x)
# print(y)


#정규화  =관계형 데이터베이스의 설계에서 중복을 최소화하게 데이터를 구조화하는 프로세스
#정규화된 데이터이다. **** 나중에 가장 중요

#print(datasets)

# print(datasets.feature_names)
# 'CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
# 'B' 'LSTAT'

# print(datasets.DESCR)    #DESCR묘사하다********


# Number of Instances: 506  예시가 506개다
# Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.
# 열이 13개이다.
# MEDV     Median value of owner-occupied homes in $1000's <<< y 값. 집값
# x는 열이 13개, y는 집값

# print(x.shape, y.shape)


############ [실습] ##############
# 1. train 0.7
# 2. R2 0.8 이상
##################################



#1. 데이터
datasets = load_boston()
x = datasets.data     #사이킷런에서 제공함
y = datasets.target   # 일반적으로 y는 target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    shuffle=True,
    train_size=0.7,
    random_state=66)



#2. 모델구성


model=Sequential()
model.add(Dense(128, input_dim=13))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mse',optimizer='adam')
model.fit(x_train, y_train, epochs=10000, batch_size=16)

#4. 평가, 예측

loss= model.evaluate(x_test, y_test)
print("loss :", loss)


y_predict = model.predict(x_test)


# R2= 결정 계수

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ',r2)


#r2스코어 :  0.6370608342321689
#r2스코어 :  0.7113113218727902
#r2스코어 :  0.7502359363945628 랜덤 80 에포 2000 배치 100
#r2스코어 :  0.7540105267214479 랜덤 80 에포 10000 배치 100
#r2스코어 :  0.7651587162346822 랜 80 에포 1000 배치 5
#r2스코어 :  0.7439550907974968 배치 3
#r2스코어 :  0.7497316969187722 배치 5 에포 1000 레이어2줄 추가
#r2스코어 :  0.7909375276777355 relu 삽입


#*****************r2스코어 :  0.8514976704615949*************** 에포 10000 배치 16 히든 레이어 128 64 32 1
#r2스코어 :  0.834079973976749
#r2스코어 :  0.7993677692568495
#r2스코어 :  0.7991566028953964
# r2스코어 :  0.7878729224562268
# loss: 17.38869285583496
#r2스코어: 0.7895267103967221
#r2스코어 :  0.7905855034072531
#r2스코어 :  0.7983256217071039
#r2스코어 :  0.7998648262557375

#r2스코어 :  0.8026022572598184

#r2스코어 : 0.799694294736808