import numpy as np
import matplotlib.pyplot as plt

def PRELU(x, alpha):
    return np.where(x < 0, alpha * (np.exp(x) - 1), x)

PRELU2 = lambda x, alpha: np.where(x < 0, alpha * (np.exp(x) - 1), x)

x = np.arange(-5, 5, 0.1)
alpha = 0.5 
y = PRELU2(x, alpha)

plt.plot(x, y)
plt.grid()
plt.show()