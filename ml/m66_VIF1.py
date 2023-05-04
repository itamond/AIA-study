import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 다중공선성
# 컬런의 상관관계에 관한 이야기
# 상관관계가 너무 높으면 제거하거나 차원 축소하는것이 나을지도 모른다
# 코릴레이션(corr)과 비슷한것
# 다중공선성 확인을 위해서는 스케일링을 먼저 해준다.(컬런이 다르면 값들의 차이가 크기 때문에)

data = {'size' : [30, 35, 40, 45, 50, 45],
        'rooms': [2, 2, 3, 3, 4, 3],
        'window': [2, 2, 3, 3, 4, 3],
        'year' : [2010, 2015, 2010, 2015, 2010, 2014],
        'price': [1.5, 1.8, 2.0, 2.2, 2.5, 2.3]
        }

df = pd.DataFrame(data)

# print(df)

x = df[['size', 'rooms','window', 'year']]
y = df['price']
scaler = StandardScaler()# 다중공선성은 통상 스탠다드 스케일러 사용한다
x_scaled = scaler.fit_transform(x)
print(x_scaled)

vif = pd.DataFrame()
vif['variables']=x.columns #컬런 이름 넣기. 3개

        # aaa for i in range(x_scaled.shape[1])  #포문의 내용이 aaa에 들어간다
vif['VIF'] = [variance_inflation_factor(x_scaled, i) for i in range(x_scaled.shape[1])] #vif의 VIF에는 포문의 반환값 세개가 들어가있다.

#   variables         VIF #다중공선성 출력하기
# 0      size  378.444444
# 1     rooms  406.111111
# 2      year   53.333333
#다중공선성이 10 이하일경우 높지 않다고 판단한다.
#현 데이터는 다중공선성이 너무 높다. 
#이럴경우 가장 높은 컬런부터 제거해본다.

print('=====================rooms 제거전=======================')
print(vif)

lr = LinearRegression()
lr.fit(x_scaled, y)
y_pred = lr.predict(x_scaled)
r2 = r2_score(y, y_pred)
print('rooms 제거 전 r2 :', r2)

# r2 : 0.9938931297709924


print('=====================rooms 제거후=======================')

x_scaled=df[['size','window', 'year']]

vif2 = pd.DataFrame()
vif2['variables']=x_scaled.columns #컬런 이름 넣기. 3개
vif2['VIF'] = [variance_inflation_factor(x_scaled, i) for i in range(x_scaled.shape[1])] #vif의 VIF에는 포문의 반환값 세개가 들어가있다.
print(vif2)

lr = LinearRegression()
lr.fit(x_scaled, y)
y_pred = lr.predict(x_scaled)
r2 = r2_score(y, y_pred)
print('rooms 제거 후 r2 :', r2)


# =====================rooms 제거전=======================
#   variables         VIF
# 0      size  378.444444
# 1     rooms         inf
# 2    window         inf
# 3      year   53.333333
# rooms 제거 전 r2 : 0.9938931297709924
# =====================rooms 제거후=======================
#   variables         VIF
# 0      size  295.182375
# 1    window  139.509263
# 2      year   56.881874
# rooms 제거 후 r2 : 0.9938931297709924