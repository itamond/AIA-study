# x는 1개
# y는 3개

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



#1. 데이터
x = np.array([range(10)])
print(x.shape) # (3, 10)
x = x.T  #(10,3)

y = np.array([[1,2,3,4,5,6,7,8,9,10], 
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
              [9,8,7,6,5,4,3,2,1,0]]) # (3, 10)

y = y.T # (10, 3)


x_train= x[:7]
x_test= x[7:]
y_train=y[:7]
y_test=y[7:]


print('x트레인',x_train)
print('x테스트',x_test)
print('y트레인',y_train)
print('y테스트',y_test)
