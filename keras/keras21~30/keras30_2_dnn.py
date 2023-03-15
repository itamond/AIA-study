from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model=Sequential()
model.add(Dense(10, input_shape=(3,)))   #(batch_size, input_dim)
model.add(Dense(units=15, ))             #출력 (batch_size, units)
model.summary()

# =================================================================
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 10)                40           # None 은 batch_size
# =================================================================
# Total params: 40
# Trainable params: 40
# Non-trainable params: 0
# _________________________________________________________________


# DNN에서의 dense 데이터 구조
# (batch_size, input_dim).  input
# (batch_size, units).      output