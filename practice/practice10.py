#사이킷런의 디아벳 데이터 셋으로 모델을 구성해보고 시각화해보자.

import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 1.데이터

datasets = load_diabetes()

x= datasets.data     #(442,10)
y= datasets.target   #(442,)

# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)
"""
  :Number of Instances: 442

  :Number of Attributes: First 10 columns are numeric predictive values

  :Target: Column 11 is a quantitative measure of disease progression one year after baseline

  :Attribute Information:
      - age     age in years
      - sex
      - bmi     body mass index
      - bp      average blood pressure
      - s1      tc, total serum cholesterol
      - s2      ldl, low-density lipoproteins
      - s3      hdl, high-density lipoproteins
      - s4      tch, total cholesterol / HDL
      - s5      ltg, possibly log of serum triglycerides level
      - s6      glu, blood sugar level
"""

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    shuffle=True, 
    train_size=0.9, 
    random_state=335)


#2. 모델구성

model=Sequential()
model.add(Dense(10, input_dim=10))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print("loss :", loss)

y_predict=model.predict(x_test)

r2=r2_score(y_test, y_predict)
print("r2 :", r2)


# plt.plot(x, y_predict, color='blue')
# plt.show()

#r2 : 0.7393734544256141