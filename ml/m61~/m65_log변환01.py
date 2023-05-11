import numpy as np
import matplotlib.pyplot as plt


data = np.random.exponential(scale=2.0, size=1000)   #exponential 기하급수적

#로그변환
log_data = np.log(data)   #로그는 넘파이에서 제공함

# 원본 데이터 히스토그램
plt.subplot(1, 2, 1)
plt.hist(data, bins=50, color='blue', alpha=0.5)
plt.title('Original')


plt.subplot(1, 2, 2)
plt.hist(log_data, bins=50, color='red', alpha=0.5)
plt.title('Log Transformed Data')
plt.show()

#로그변환하면 정규분포에 가까운 형태로 변한다.
#데이터의 분포를 얼마나 가운데로 모으느냐가 개발자의 실력