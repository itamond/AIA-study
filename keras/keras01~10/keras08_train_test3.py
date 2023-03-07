#넘파이 리스트 셔플 슬라이싱
import sklearn
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])     #=  뒤에 콤마를 찍는다고 해서 문제가 되진 않는다.


# [검색]  train과 test를 섞어서 7:3으로 찾을 수 있는 방법!!
# 힌트 사이킷런


#shuffle=false


# shuffle = True
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.3, 
    shuffle=True, #섞겠다.
    random_state=1004
    )
#random_state - 랜덤 시드값 .. 랜덤이라도 다음에 같은 환경에서 훈련할 수 있도록 잡아주는 시드값 ex) 난수값 1004로 랜덤하게 뽑으라는 뜻
#test_size, train_size 무엇으로 해도 상관 없다.

print('X_train shape:', x_train.shape)

print('X_test shape:', x_test.shape)

print('y_train shape:', y_train.shape)

print('y_test shape:', y_test.shape)



print('X_train :', x_train)

print('X_test :', x_test)

print('y_train :', y_train)

print('y_test :', y_test)

#2. 모델 구성
model=Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)


#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss :',loss)
result = model.predict([11])
print('[11]의 예측값 :', result)