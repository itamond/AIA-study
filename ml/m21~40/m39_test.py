import numpy as np
import matplotlib.pyplot as plt

aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
                [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]])
aaa = np.transpose(aaa)

def outliers(data_out):
    for i in range(data_out.shape[1]):
        quartile_1, q2, quartile_3 = np.percentile(data_out[:,i], [25, 50, 75])
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        outlier_indices = np.where((data_out[:,i] > upper_bound) | (data_out[:,i] < lower_bound))[0]
        whereoi = np.where((data_out[:,i] > upper_bound) | (data_out[:,i] < lower_bound))
        if outlier_indices.size > 0:
            print(i+1,"번째 컬런의 이상치 :", data_out[outlier_indices,i],'\n',' 이상치의 위치 :', whereoi[0])
        else:
            print(i+1,"번째 컬런 이상치 없음")

bbb=outliers(aaa)
print(bbb)
# plt.boxplot(aaa)
# # plt.boxplot(bbb)
# plt.show()