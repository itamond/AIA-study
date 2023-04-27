import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.experimental import enable_iterative_imputer 
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                    [2, 4, np.nan, 8, np.nan],
                    [2, 4, 6, 8, 10],
                    [np.nan, 4, np.nan, 8, np.nan]]
                    ).transpose()

print(data)
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)


from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer #결측치에 대한 책임을 돌린다
#KNN = 최근접 이웃

# imputer = SimpleImputer() #디폴트는 mean.
# imputer = SimpleImputer(strategy='mean') #디폴트는 mean.
# imputer = SimpleImputer(strategy='median')  #중위값
# imputer = SimpleImputer(strategy='most_frequent')  #최빈값 #제일 많이 등장한 값. 갯수가 같을 경우에 가장 작은 값
# imputer = SimpleImputer(strategy='constant') # 0으로 채우기
# imputer = SimpleImputer(strategy='constant', fill_value=77) #지정한 값으로 채우기
# imputer = KNNImputer() #KNN 알고리즘으로 평균값 찾음
# imputer = IterativeImputer() #선형회귀 느낌으로 값을 찾음
# imputer = IterativeImputer(estimator=DecisionTreeRegressor()) 
imputer = IterativeImputer(estimator=XGBRegressor())

data2 = imputer.fit_transform(data)
print(data2)

