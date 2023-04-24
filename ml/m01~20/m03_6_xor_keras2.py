#xor = 같으면 0, 다르면 1이다. 0과 0이면 0 0과 1이면 1

import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]] # (4,2)
y_data = [0, 1, 1, 0]


#2. 모델
# model = SVC()

model = Sequential()
model.add(Dense(32,activation='relu',input_dim=2))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#3. 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size = 1, epochs=100)

#4. 평가 예측
y_predict = model.predict(x_data)


# result = model.score(x_data, y_data)
result = model.evaluate(x_data, y_data)
print('acc :', result[1])



acc = accuracy_score(y_data, np.round(y_predict))
print('accuracy_score :', acc)