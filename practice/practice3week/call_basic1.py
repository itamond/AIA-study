import warnings 
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


#1. 데이터
data = pd.read_csv('C:/AIA-study/_data/basic/train.csv', index_col=0)
test = pd.read_csv('C:/AIA-study/_data/basic/test.csv', index_col=0)

data = data.astype('int64')
test = test.astype('int64')


pd.set_option('display.max_columns',None)

plt.rcParams['font.family'] = 'Malgun Gothic'  

plt.style.use('ggplot')
plt.figure(figsize=(25, 20))
plt.suptitle("Data Histogram", fontsize = 40)
cols = data.columns
for i in range(len(cols)):
    plt.subplot(5, 5, i+1) 
    plt.title(cols[i], fontsize=20)
    if len(data[cols[i]].unique()) > 20: 
        plt.hist(data[cols[i]], bins=20, color='b', alpha=0.7) 
        
    else:
        temp = data[cols[i]].value_counts()
        plt.bar(temp.keys(), temp.values, width=0.5, alpha=0.7)
        plt.xticks(temp.keys())
        
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
# target = "target"
# # 범주형 데이터 분리
# categorical_feature = data.columns[data.dtypes=='object']

# plt.figure(figsize=(20,15))
# plt.suptitle("Violin Plot", fontsize=40)

# # id는 제외하고 시각화합니다.
# for i in range(len(categorical_feature)):
#     plt.subplot(2,2,i+1)
#     plt.xlabel(categorical_feature[i])
#     plt.ylabel(target)
#     sns.violinplot(x= data[categorical_feature[i]], y= data[target])
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

# 수치형 데이터 분리
numeric_feature = data.columns[(data.dtypes=='int64') | (data.dtypes=='float')]
num_data = data[numeric_feature]

# 박스플롯
fig, axes = plt.subplots(3, 6, figsize=(25, 20))

fig.suptitle('feature distributions per quality', fontsize= 40)
for ax, col in zip(axes.flat, num_data.columns[:-1]):
    sns.boxplot(x= '전화해지여부', y= col, ax=ax, data=num_data)
    ax.set_title(col, fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# data.plot(kind='box', subplots=True, layout=(5, 5), figsize=(15, 21))
# plt.show()       



def outliers_iqr(data):
    q1, q3 = np.percentile(data, [25, 75])
    # 넘파이의 값을 퍼센트로 표시해주는 함수

    iqr = q3 - q1   
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    
    return np.where((data > upper_bound) | (data < lower_bound))


Callbox_index_data = outliers_iqr(data['음성사서함이용'])[0]
Date_index_data = outliers_iqr(data['가입일'])[0]

data.loc[Callbox_index_data, '음성사서함이용'] = data['음성사서함이용'].mean()
data.loc[Date_index_data, '가입일'] = data['가입일'].mean()

data.plot(kind='box', subplots=True, layout=(5, 5), figsize=(15, 21))
plt.show()       