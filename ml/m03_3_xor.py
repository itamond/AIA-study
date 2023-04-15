#xor = 같으면 0, 다르면 1이다. 0과 0이면 0 0과 1이면 1

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron

#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0, 1, 1, 0]


#2. 모델
# model = LinearSVC()
model = Perceptron()  #단층 퍼셉트론


#3. 훈련
model.fit(x_data, y_data)

#4. 평가 예측
y_predict = model.predict(x_data)

result = model.score(x_data, y_data)

print('model.score :', result)

acc = accuracy_score(y_data, y_predict)
print('accuracy_score :', acc)