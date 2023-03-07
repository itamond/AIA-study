from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
x = np.array(range(1,17)) # 스칼라 10개가 모인 벡터 1개
y = np.array(range(1,17))

# x_val = np.array([14,15,16])    #  13개의 데이터중에 3개는 발리데이션, 3개는 테스트, 나머지는 트레인
# y_val = np.array([14,15,16])

# x_test = np.array([11,12,13])
# y_test = np.array([11,12,13])

# x_val = x_train[13:16]
# y_val = y_train[13:16]

# x_test = x_train[10:13]
# y_test = y_train[10:13]

# print(x_val, x_test)
# print(y_val, y_test)


# 실습 :: 잘라봐!!!

# 10 : 3 : 3

x_train, x_test, y_train, y_test= train_test_split(x, y, 
        test_size=0.2, random_state=123)

#2. 모델구성
model = Sequential()
model.add(Dense(5, activation='linear', input_dim=1))
model.add(Dense(10, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1,
          validation_split=0.2
          )    #자동완성 불러오기 컨트롤 + 스페이스


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

result = model.predict([17])
print('17의 예측값 :', result)



#[[16.925383]]

#loss : 2.3002474335953593e-05
# 17의 예측값 : [[17.001276]]


# loss : 1.2247861613801092e-09
# 17의 예측값 : [[16.999826]]





# print(x_train,x_val,x_test)
# print(y_train,y_val,y_test)

#