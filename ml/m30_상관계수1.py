import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# 1. 데이터
datasets = load_iris()
print(datasets.feature_names) # 판다스는 columns

x = datasets['data']
y = datasets.target

df = pd.DataFrame(x, columns=datasets.feature_names)   #columns=datasets.feature_names => 컬럼 이름 넣기
# print(df)

df['Target(Y)'] = y  #컬럼 새로 만들기
print(df)

print("==================== 상관계수 히트 맵 =====================")
print(df.corr())  #Correlation 상관성 상관관계는 무조건 신용해서는 안된다. Y와의 상관관계를 우선 확인한다.

#                    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  Target(Y)
# sepal length (cm)           1.000000         -0.117570           0.871754          0.817941   0.782561
# sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126  -0.426658
# petal length (cm)           0.871754         -0.428440           1.000000          0.962865   0.949035
# petal width (cm)            0.817941         -0.366126           0.962865          1.000000   0.956547
# Target(Y)                   0.782561         -0.426658           0.949035          0.956547   1.000000

import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)

plt.show()

