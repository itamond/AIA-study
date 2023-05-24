import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**2 -4*x + 6

x = np.linspace(-1, 6, 100)

# np.linspace 
# 인자를 3개를 기본으로 넣어주면 되는데, 구간 시작점, 구간 끝점, 구간 내 숫자 개수

print(x, len(x))

y = f(x)

plt.plot(x, y, 'k-')
plt.plot(2, 2, 'sk')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()