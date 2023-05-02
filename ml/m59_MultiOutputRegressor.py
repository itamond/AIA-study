import numpy as np
import pandas as pd
from sklearn.datasets import load_linnerud
from sklearn.linear_model import Lasso, Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error

x, y = load_linnerud(return_X_y=True)
# print(x)
# print(y)
# print(x.shape, y.shape) #(20, 3) (20, 3)


# model = Ridge()
# model.fit(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__,
#       'MAE :', 
#       np.round(mean_absolute_error(y, y_pred),3))
# print(model.predict([[2, 110, 43]])) 

# model = XGBRegressor()
# model.fit(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__,
#       'MAE :', 
#       np.round(mean_absolute_error(y, y_pred),3))
# print(model.predict([[2, 110, 43]])) 

# model = MultiOutputRegressor(LGBMRegressor())  #LGBM 의 다중 output 출력을 위해 래핑해줌.
# model.fit(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__,
#       'MAE :', 
#       np.round(mean_absolute_error(y, y_pred),3))
# print(model.predict([[2, 110, 43]])) 
# [138.  33.  68.] 예상
# [[187.32842123  37.0873515   55.40215097]] 릿지 결과 
# XGB 결과
# 스코어 : 0.9999999567184008
# [[138.00215   33.001656  67.99831 ]]

# LGBM은 1차원 데이터만 받는다... 때문에 3차원 데이터가 y인 경우 세번 훈련 해야함.
# ValueError: y should be a 1d array, got an array of shape (20, 3) instead.
# [[178.6  35.4  56.1]] 래핑해서 출력한 결과


# Ridge MAE : 7.457
# [[187.32842123  37.0873515   55.40215097]]

# XGBRegressor MAE : 0.001
# [[138.00215   33.001656  67.99831 ]]

# MultiOutputRegressor MAE : 8.91 LGBM
# [[178.6  35.4  56.1]]

# MultiOutputRegressor MAE : 0.215 CatBoost
# [[138.97756017  33.09066774  67.61547996]]

# model = MultiOutputRegressor(CatBoostRegressor(verbose=0))
model = CatBoostRegressor(loss_function='MultiRMSE')
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__,
      'MAE :', 
      np.round(mean_absolute_error(y, y_pred),3))
print(model.predict([[2, 110, 43]])) 
# CatBoostRegressor MAE : 0.064
# [[138.21649371  32.99740595  67.8741709 ]]
