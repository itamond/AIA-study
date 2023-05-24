import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # 1 / 1+ e의 -x승     -> 모든 값이 0과 1 사이로 만듬

sigmoid = lambda x : 1 / (1 + np.exp(-x))

x = np.arange(-5, 5, 0.1)  #np.arange(시작점(생략 시 0), 끝점(미포함), step size(생략 시 1), 실수 단위도 가능한 range
print(x) 
print(len(x))

y = sigmoid(x)
plt.plot(x,y)
plt.grid()
plt.show()
