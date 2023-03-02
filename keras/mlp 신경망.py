# Sequential Model
# layer 를 계속 추가할 수 있다.
# 직접 넣는 방법

import tensorflow as tf

model = tf.keras.Sequential ([
    tf.keras.layers.Dense(units=4, input_shape=(3,)),
    tf.keras.layers.Dense(units=4),
    tf.keras.layers.Dense(units=1)
])

model.summary()
