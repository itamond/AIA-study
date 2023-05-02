import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# x = np.arange(8).reshape(4,2)
# print(x)
# # [[0 1]
# #  [2 3]
# #  [4 5]
# #  [6 7]]

# pf = PolynomialFeatures(degree=2)
# x_pf = pf.fit_transform(x)

# print(x_pf)
# print(x_pf.shape)

# # [[ 1.  0.  1.  0.  0.  1.]
# #  [ 1.  2.  3.  4.  6.  9.]
# #  [ 1.  4.  5. 16. 20. 25.]
# #  [ 1.  6.  7. 36. 42. 49.]]
# # (4, 6)

print("================================================")
x = np.arange(8).reshape(4,2)
print(x)
# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]

pf = PolynomialFeatures(degree=3)
x_pf = pf.fit_transform(x)

print(x_pf)
print(x_pf.shape)

# [[ 1.  0.  1.  0.  0.  1.]
#  [ 1.  2.  3.  4.  6.  9.]
#  [ 1.  4.  5. 16. 20. 25.]
#  [ 1.  6.  7. 36. 42. 49.]]
# (4, 6)
