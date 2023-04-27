# import numpy as np
# from sklearn.covariance import EllipticEnvelope
# aaa = np.array([[-10,2,3,4,5,6,700,8,9,10,11,12,50],
#                [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]])
# # aaa = aaa.reshape(-1, 1)
# aaa = np.transpose(aaa)


# outliers = EllipticEnvelope(contamination=.1) #contamination = 이상치 범위 지정 파라미터
# outliers.fit(aaa)
# results = outliers.predict(aaa)
# print(results)
# # [ 1  1  1  1  1  1 -1  1  1  1  1  1 -1]   -1은 이상치이다.




import numpy as np
from sklearn.covariance import EllipticEnvelope

aaa = np.array([[-10,2,3,4,5,6,700,8,9,10,11,12,50],
               [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]])

outliers = EllipticEnvelope(contamination=.1)

for i, column in enumerate(aaa): 
    outliers.fit(column.reshape(-1, 1))  
    results = outliers.predict(column.reshape(-1, 1))
    outliers_save = np.where(results == -1)[0]
    # print(outliers_save)
    # print(outliers_save[0])
    outliers_values = column[outliers_save] 
    
    print(f"{i+1}번째 컬런의 이상치 : {', '.join(map(str, outliers_values))}\n 이상치의 위치 : {', '.join(map(str, outliers_save))}")
    
    
    
    