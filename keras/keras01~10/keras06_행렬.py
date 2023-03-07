import numpy as np

x1 = np.array([[1,2],[3,4]])    # 2X2 2행 2열
x2 = np.array([[[1,2,3]]])    # 1x1x3 1행 3열  1개
x3 = np.array([[[1,2,3], [4,5,6]]])  #  1x2x3 2행 3열 1개
x4 = np.array([[1], [2], [3]]) #    1x3x1   3행 1열 1개
x5 = np.array([[[1]], [[2]], [[3]]])  # 3x1x1 1행 1열 3개
x6 = np.array([[[1,2], [3,4]], [[5,6], [7,8]]])  # 2x2x2 2행 2열 2개
x7 = np.array([[[1,2]], [[3,4]], [[5,6]], [[7,8]]])  # 4x1x2 1행 2열 4개


print(x1.shape) # (2, 2)
print(x2.shape) # (1, 1, 3)
print(x3.shape) # (1, 2, 3)
print(x4.shape) # (3, 1)
print(x5.shape) # (3, 1, 1)
print(x6.shape) # (2, 2, 2)
print(x7.shape) # (4, 1, 2)
