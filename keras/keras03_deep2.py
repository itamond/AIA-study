# 1. 데이터
import numpy as np
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

# 2. 모델구성
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 

model = Sequential()
model.add(Dense(3, input_dim=1)) 
model.add(Dense(4)) 
model.add(Dense(5)) 
model.add(Dense(7)) 
model.add(Dense(9)) 
model.add(Dense(11)) 
model.add(Dense(13)) 
model.add(Dense(11)) 
model.add(Dense(10)) 
model.add(Dense(9)) 
model.add(Dense(8)) 
model.add(Dense(6)) 
model.add(Dense(5)) 
model.add(Dense(3))
model.add(Dense(1)) 

# 3. 컴파일, 훈련   
model.compile(loss='mse', optimizer='adam') 
model.fit(x, y, epochs=200)  

# 4. 평가, 예측
loss = model.evaluate(x, y) # evaluate= 평가 함수. fit에서 생성된 w 값에 x와 y데이터를 넣어서 판단을 하는것이다. 지금의 경우는 너무나 정제된  x와 y데이터를 넣는 예시.
print('loss : ', loss)

result=model.predict([4]) #predict= 예측 함수 위에서 []라는 데이터를 넣었으므로 여기서도 []를 넣는다

print("[4]의 예측값 :", result)



