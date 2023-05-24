import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG16


# model = VGG16() # include_top = True, input_shape = (224, 224, 3), output layer 1000개
model = VGG16(weights='imagenet', include_top=False,
              input_shape=(32, 32, 3))

#include_top을 False로 하면 모델의 윗부분만 가져오고, 튜닝 가능하게 한다. 커스터마이징을 편하게 만들어줌.

model.summary()

print(len(model.weights))  # 32 include_top False -> 26
print(len(model.trainable_weights))  # 32 include_top False -> 26



# include_top = True
# 1. 기존 모델의 FC layer를 사용한다.
# 2. input_shape=(224, 224, 3) 고정값. 변환 불가

# input_1 (InputLayer)        [(None, 224, 224, 3)]     0

#  flatten (Flatten)           (None, 25088)             0
#  fc1 (Dense)                 (None, 4096)              102764544
#  fc2 (Dense)                 (None, 4096)              16781312
#  predictions (Dense)         (None, 1000)              4097000
# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0



# include_top = False
# 1. 기존 모델의 FC layer 삭제, 커스터마이징
# 2. input_shape 변경 가능.

#  input_1 (InputLayer)        [(None, 32, 32, 3)]       0
# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0