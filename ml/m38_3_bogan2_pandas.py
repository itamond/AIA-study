import pandas as pd
import numpy as np

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                    [2, 4, np.nan, 8, np.nan],
                    [2, 4, 6, 8, 10],
                    [np.nan, 4, np.nan, 8, np.nan]]
                    ).transpose()

print(data)
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)


#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

# 0. 결측치 확인
print(data.isnull())
print(data.isnull().sum())   #True가 결측치
print(data.info())   #info와 DESCR 활용하여 데이터 확인하는 습관을 기르자.

# dtype: int64
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 5 entries, 0 to 4
# Data columns (total 4 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   x1      4 non-null      float64
#  1   x2      3 non-null      float64
#  2   x3      5 non-null      float64
#  3   x4      2 non-null      float64
# dtypes: float64(4)
# memory usage: 288.0 bytes
# None


# 1. 결측치 삭제
print("===================== 결측치 삭제 ====================")
# print(data['x1'].dropna())   # 이렇게하면 그 열에서만 삭제되기 때문에 의미가 없다.
# print(data.dropna()) # 디폴트가 행 위주 삭제.
print(data.dropna(axis=0)) # 행 위주 삭제 / 디폴트
print(data.dropna(axis=1)) # 열 위주 삭제


#2-1. 특정값 - 평균
print("=================== 결측치 처리 mean() ================")
# data.mean()  #각 컬런별 평균을 뽑아줌
means = data.mean() 
print('평균 : ', means)
data2 = data.fillna(means)
print(data2)

#2-2. 특정값 - 중위값
print('=================== 결측치 처리 median() ==============')
median = data.median()
print('중위값 : ', median)
data3 = data.fillna(median)
print(data3)

#2-3. 특정값 - ffill, bfill
print('=================== 결측치 처리 ffill, bfill ==============')
data4 = data.fillna(method='ffill')
print(data4)    #앞의 값이 없는 경우 값을 가져올 수 없다.
data5 = data.fillna(method='bfill')
print(data5)    #뒤의 값이 없는 경우 값을 가져올 수 없다.

#2-4. 특정값 - 임의값으로 채우기
print('=================== 결측치 처리 - 임의의 값으로 채우기 ==============')
# data6 = data.fillna(777777)
data6 = data.fillna(value='777777')
print(data6)


##########################특정칼럼만##############################

#1. x1컬럼에 평균값
means = data['x1'].mean() #평균값 저장
data['x1'] = data['x1'].fillna(means) #means로 채우기
print('fill_na_x1_means :','\n',data)


#2. x2컬럼에 중위값
medians = data['x2'].median() #중위값 저장
data['x2'] = data['x2'].fillna(medians) #medians로 채우기
print('fill_na_x2_median :','\n',data)


#3. x4컬럼에 ffill한 후 / bfill
data = data.fillna(method='ffill') #ffill 적용
data = data.fillna(method='bfill') #bfill 적용
print('fill_na_x4_fbfill :','\n',data)


#머신러닝은 다차원 y값(y가 두개 이상인것) 못뽑는다.
