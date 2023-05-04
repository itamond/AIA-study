from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
#1. 데이터셋
datasets = fetch_california_housing()
df = pd.DataFrame(datasets.data, columns = datasets.feature_names)
df['target'] = datasets.target

print(df)

# df.boxplot()
# df.plot.box()
# plt.show()

df.info()
print(df.describe())

# df['Population'].boxplot() 안먹힌다
# df['Population'].plot.box()
# plt.show()

# df['Population'].hist(bins=50) #히스토그램 그리기
# plt.show()

df['target'].hist(bins=50) #bins=50은 분위수를 50개씩 잘랐다는 소리이다.
# plt.show()

y = df['target']
x = df.drop(['target'], axis=1)

x.hist(bins=50) #bins=50은 분위수를 50개씩 잘랐다는 소리이다.

plt.show()

################# x population 로그 변환 ######################
# x['Population'] = np.log(x['Population']) #로그의 치명적인 단점은 0을 계산할 수 없는 것이다.
x['Population'] = np.log1p(x['Population']) #때문에 log1p를 사용함. 값을 1을 더하는것? 후에 지수변환 할때는 exp1m (exp 1마이너스)
#분류형은 로그변환불가

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=337
)

#################y로그변환
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)
#########################

x.hist(bins=50) #bins=50은 분위수를 50개씩 잘랐다는 소리이다.

plt.show()

# 2. model
model = RandomForestRegressor(random_state=337)

# 3. 컴파일,훈련
model.fit(x_train,y_train)


# 4. 평가, 예측
score = model.score(x_test, y_test)
r2 = r2_score(np.expm1(y_test), np.expm1(model.predict(x_test))) #로그변환된 값을 다시 지수변환 하고 비교해야한다

print('r2 :', r2)

print('score :', score)

# 로그변환 전
# score : 0.8021183994602941

# x[Population] 로그변환 후
# score : 0.8022669492389668

# y 만 로그변환 후
# score : 0.8244322268075517

# x y 로그변환 후
# score : 0.824475366593445

#데이터의 히스토그램을 파악해서 log 변환을 적용해주면 좋다.

