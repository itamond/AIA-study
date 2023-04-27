import numpy as np
from sklearn.covariance import EllipticEnvelope
aaa = np.array([-10,2,3,4,5,6,700,8,9,10,11,12,50])
aaa = aaa.reshape(-1, 1)


outliers = EllipticEnvelope(contamination=.3) #contamination = 이상치 범위 지정 파라미터
outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)
# [ 1  1  1  1  1  1 -1  1  1  1  1  1 -1]   -1은 이상치이다.

