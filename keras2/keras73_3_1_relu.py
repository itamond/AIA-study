import numpy as np
import matplotlib.pyplot as plt


def relu(x) :
    return np.maximum(0, x)     #0과 x를 비교하여 큰 값을 뽑는 함수이다. 따라서 0보다 작으면 0으로 출력됨.

relu2 = lambda x : np.maximum(0, x)

x = np.arange(-5, 5, 0.1)
y = relu2(x)

plt.plot(x, y)
plt.grid()
plt.show()