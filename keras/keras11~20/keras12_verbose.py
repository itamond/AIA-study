#verbose


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score



#1. 데이터
datasets = load_boston()
x = datasets.data   
y = datasets.target 

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
model.fit(x_train, y_train, epochs=10, batch_size=2,verbose='auto') 
#verbose 0 ->훈련과정 삭제
#verbose의 디폴트는 1이다. 다 보여준다.
#verbose 2는 진행(프로그레스)바가 없다.
#verbose 나머지는 에포만 나온다

# #4. 평가, 예측

# loss= model.evaluate(x_test, y_test, verbose=0)
# print("loss :", loss)


y_predict = model.predict(x_test)



r2 = r2_score(y_test, y_predict)
print('r2스코어 : ',r2)


