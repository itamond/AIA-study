import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler
import matplotlib.pyplot as plt
import matplotlib

x,y=make_blobs(
    n_samples=50,     #데이터가 50개로 줄어든다
    centers=2, #클러스터(군집)의 중심점 갯수. 즉 클러스터의 갯수 / y의 라벨이라고 할수도 있다.
    cluster_std = 1, #클러스터의 표준편차
    random_state=337
)

#가우시안 정규분포 기준으로 만든 샘플

plt.rcParams['font.family'] = 'Malgun Gothic'           #폰트 변환하는 두가지 방법

fig, ax = plt.subplots(2, 2, figsize=(12,8))

ax[0,0].scatter(x[:,0], #모든 행의 0번재 열
            x[:,1],
            c=y,
            edgecolors='black')

ax[0,0].set_title('오리지날')

# plt.show()


scaler = QuantileTransformer(n_quantiles=50)
x_trans = scaler.fit_transform(x)


ax[0,1].scatter(x_trans[:,0], #모든 행의 0번재 열
            x_trans[:,1],
            c=y,
            edgecolors='black')
ax[0,1].set_title(type(scaler).__name__)

scaler = PowerTransformer()
x_trans = scaler.fit_transform(x)


ax[1,0].scatter(x_trans[:,0], #모든 행의 0번재 열
            x_trans[:,1],
            c=y,
            edgecolors='black')
ax[1,0].set_title(type(scaler).__name__)


scaler = StandardScaler()
x_trans = scaler.fit_transform(x)


ax[1,1].scatter(x_trans[:,0], #모든 행의 0번재 열
            x_trans[:,1],
            c=y,
            edgecolors='black')

ax[1,1].set_title(type(scaler).__name__)




plt.show()
