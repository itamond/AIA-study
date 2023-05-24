import numpy as np
import matplotlib.pyplot as plt


def softmax(x) :
    return np.exp(x) / np.sum(np.exp(x))   #e의 x^1 / e의 x^1 + e의  x^2 +.... 따라서 모든 e의 x^n 의 합은 1

softmax2 = lambda x : np.exp(x) / np.sum(np.exp(x))

x = np.arange(1, 5)
y = softmax2(x)

ratio = y
labels = y

plt.pie(ratio, labels, shadow=True, startangle=90)
plt.show()