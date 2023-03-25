import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping
#1. 데이터

datasets = np.array([1,2,3,4,5,6,7,8,9,10])

# y = ?

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]])
#y값을 지정해줘야하기 때문에 10은 뺀다

y = np.array([4, 5, 6, 7, 8, 9, 10])

print(x.shape, y.shape)  #(7, 3) (7,)


#RNN은 통상 3차원 데이터로 훈련.
#[1,2,3] 훈련을 한다면 1한번 2한번 3한번 훈련한다

#x의 shape = (행, 열, 몇개씩 훈련하는지) 
#이번 경우에는 1개씩 훈련하기때문에 끝을 1로 리쉐이프 해줘야함.

#LSTM의 게이트는 3개, 스테이트 1개  -> 인풋, 아웃풋, 포겟 게이트, 셀 스테이트

#LSTM 속도 느림, 연산량 4배 많음, 성능은 좋음

x = x.reshape(7, 3, 1)
print(x)

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(64, input_shape=(3, 1)))        #input_shape = 행 빼고 나머지
model.add(GRU(64, input_shape=(3, 1)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss',
                   patience=50,
                   verbose=1,
                   mode='auto',
                   restore_best_weights=True)
model.fit(x,y, epochs=5000, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x, y)
x_predict = np.array([8,9,10]).reshape(1,3,1)         #인풋 쉐이프와 모양을 맞춰주는 reshape  #[[[8],[9],[10]]]
# print(x_predict.shape) #(1, 3, 1)

result = model.predict(x_predict)
print('loss :', loss)
print('[8,9,10]의 결과 : ', result)


# [8,9,10]의 결과 :  [[10.808131]]
# [8,9,10]의 결과 :  [[10.8569145]]

# [8,9,10]의 결과 :  [[10.935052]]