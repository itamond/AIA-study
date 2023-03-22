import numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)    #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)      #(10000, 28, 28) (10000,)


#3차원 데이터이기 때문에 cnn을 하기 위해서는 4차원 데이터, 60000, 28, 28, 1 로 reshape 해줘야한다.

print(x_train[0])
print(y_train[39085])

import matplotlib.pyplot as plt
plt.imshow(x_train[39085], 'gray')           #그림 그려주는 함수
plt.show()