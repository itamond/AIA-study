from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler


#1. 데이터셋
datasets = fetch_california_housing()
df = pd.DataFrame(datasets.data, columns = datasets.feature_names)
df['target'] = datasets.target



y = df['target']
x = df.drop(['target'], axis=1)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#다중공선성  반드시 스케일링 먼저 해준다.(통상 스탠다드 스케일러 사용한다.)

#학자들은 5 이하가 좋다고 보고, 개발자들은 10 이하로 본다

vif = pd.DataFrame()
vif['variables']=x.columns #컬런 이름 넣기. 3개

        # aaa for i in range(x_scaled.shape[1])  #포문의 내용이 aaa에 들어간다
vif['VIF'] = [variance_inflation_factor(x_scaled, i) for i in range(x_scaled.shape[1])] #vif의 VIF에는 포문의 반환값 세개가 들어가있다.

print(vif)

#VIF의 원칙
#1. scaling한다.
#2. y를 넣지 않는다.

#     variables       VIF
# 0      MedInc  2.501295
# 1    HouseAge  1.241254
# 2    AveRooms  8.342786
# 3   AveBedrms  6.994995
# 4  Population  1.138125
# 5    AveOccup  1.008324
# 6    Latitude  9.297624
# 7   Longitude  8.962263

# 방법
# 1. Latitude 를 삭제한다.
# 2. Longitude를 삭제한다.
# 3. Latitude와 Longitude를 PCA 한다

#상단의 스케일 x를 사용하면 안된다.
#상단의 스케일x는 다중공선성 확인을 위한것.
#때문에 트레인 테스트 분리 후 스케일을 다시한다.



x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=337, test_size=0.2,
)
scaler2 = StandardScaler()
x_train = scaler2.fit_transform(x_train)
x_test = scaler2.transform(x_test)

#2. 모델
model = RandomForestRegressor(random_state=337)

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('======================컬런 삭제 전=======================')
print('결과 :', results)

# 결과 : 0.8020989209821241
print('======================컬럼 삭제 후=======================')


high_vif_cols = vif[vif['VIF'] >= 5]['variables']

# Drop high VIF columns one by one using a for loop
for col in high_vif_cols:
    x = x.drop([col], axis=1)



x = x.drop(['Latitude','Longitude'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=337, test_size=0.2,
)


scaler2 = StandardScaler()
x_train = scaler2.fit_transform(x_train)
x_test = scaler2.transform(x_test)

model.fit(x_train,y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('결과 :', results)


# ======================컬런 삭제 전=======================
# 결과 : 0.8020989209821241
# ======================컬런 삭제 후=======================
# 결과 : 0.7298013684941245

#Longitude 추가 삭제 후
#결과 : 0.6756589579712217