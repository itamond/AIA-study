import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])

y = np.array([[1,2,3,4,5,6,7,8,9,10]]) # (1, 10)


print (x)

print (y)
