import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imputer = IterativeImputer(XGBRegressor())

a = np.array([[[1,2], [3,4]], [[5,6], [7,8]],[[9,None], [11,12]]])
print(a.shape)


for i in range(a.shape[0]):
    a[i] = imputer.fit_transform(a[i].reshape(a.shape[1], a.shape[2]))

print(a)