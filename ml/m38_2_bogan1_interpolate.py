import numpy as np
import pandas as pd
from datetime import datetime

dates = ['4/25/2023', '4/26/2023', '4/27/2023', '4/28/2023', '4/29/2023', '4/30/2023']
dates = pd.to_datetime(dates)
print(dates)
print(type(dates))

print("======================================")
ts = pd.Series([2, np.nan, np.nan, 8, 10, np.nan], index=dates)    #판다스의 벡터개념. 1차원의 데이터 집합. 열과 컬런으로 볼 수 있다. 시리즈가 모이면 dataframe이 된다.
print(ts)
print("======================================")

ts = ts.interpolate() #판다스에서 제공하는 보간
print(ts)

# ======================================
# 2023-04-25     2.0
# 2023-04-26     4.0
# 2023-04-27     6.0
# 2023-04-28     8.0
# 2023-04-29    10.0
# 2023-04-30    10.0 끝의 값은 이전값과 선을 그을 포인트가 없어서 ffill 됨





