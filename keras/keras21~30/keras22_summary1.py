import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense



#1. 데이터
x= np.array([1,2,3])
y= np.array([1,2,3])

#2. 모델구성
model =Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

model.summary()


# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 5)                 10              5+5     param= 노드와 노드의 계산 수 + bias 계산 수
# _________________________________________________________________
# dense_1 (Dense)              (None, 4)                 24              20+4
# _________________________________________________________________
# dense_2 (Dense)              (None, 3)                 15              12+3
# _________________________________________________________________
# dense_3 (Dense)              (None, 2)                 8               6+2
# _________________________________________________________________
# dense_4 (Dense)              (None, 1)                 3               2_1
# =================================================================
# Total params: 60
# Trainable params: 60
# Non-trainable params: 0
# _________________________________________________________________