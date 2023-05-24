import numpy as np
import matplotlib.pyplot as plt

def Thresholded_ReLU(x, theta):
    return np.where(x >= theta, x, 0)

Thresholded_ReLU2 = lambda x, theta: np.where(x >= theta, x, 0)

x = np.arange(-5, 5, 0.1)
theta = 0.5  
y = Thresholded_ReLU2(x, theta)

plt.plot(x, y)
plt.grid()
plt.show()