import numpy as np
aaa = np.array([-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50])

def outliers(data_out):
    quartile_1, q2, quartile_3 =np.percentile(data_out, [25, 50, 75])
    print("1사분위 : ", quartile_1) # 4
    print("q2 : ", q2) # 7
    print("3사분위 : ", quartile_3) # 10
    iqr = quartile_3 - quartile_1  #6
    print("iqr : ", iqr)
    lower_bound = quartile_1 - (iqr * 1.5) # -5
    upper_bound = quartile_3 + (iqr * 1.5) # 19
    
    return np.where((data_out>upper_bound) | (data_out<lower_bound)) #where = if와 비슷한 개념. 입력한 값을 배출한다

# IQR 의 1.5배수를 하는 이유는 중위 범위의 50% 안팍의 데이터는 정상 값으로 본다는 의미이다.
# 따라서 데이터마다 다른 배수값을 지정할수도 있다.

    
outliers_loc = outliers(aaa)
print("이상치의 위치 : ", outliers_loc)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()