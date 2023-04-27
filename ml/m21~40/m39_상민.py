import numpy as np

aaa = ([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
       [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]])

aaa = np.transpose(aaa)

#실습 outlier1을 이용해서 이상치를 찾아라!

def outilers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,[25,50,75], axis=0)
    print("1사분위 : ", quartile_1) 
    print("q2 :", q2)               
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 - quartile_1
    print("iqr : ", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)  
    upper_bound = quartile_3 + (iqr * 1.5)  
    outliers = np.where((data_out > upper_bound) | (data_out < lower_bound))
    return list(zip(outliers[0], outliers[1])) #outliers[0],outliers[1]-> 1차원 배열

outilers_loc = outilers(aaa)
print("이상치의 위치 : ", outilers_loc)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()