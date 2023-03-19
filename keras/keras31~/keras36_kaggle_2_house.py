import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

plt.style.use('seaborn')
sns.set(font_scale=1.5)

import warnings
warnings.filterwarnings("ignore")





#1. 데이터
os.listdir("./_data/house_price/")


df_train = pd.read_csv('./_data/house_price/train.csv')
df_test = pd.read_csv('./_data/house_price/test.csv')


# print(df_train.head())

# print(df_train.shape, df_test.shape)    #(1460, 80) (1459, 79)


numerical_feats = df_train.dtypes[df_train.dtypes !='object'].index
# print('숫자형 피쳐 :', len(numerical_feats))

categorical_feats = df_train.dtypes[df_train.dtypes =='object'].index
# print('범주형 피쳐 :', len(categorical_feats))

# 숫자형 피쳐 : 37
# 범주형 피쳐 : 43


# print(df_train[numerical_feats].columns)
# print('*'*80)
# print(df_train[categorical_feats].columns)


# Index(['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
#        'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
#        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
#        'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
#        'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
#        'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
#        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
#        'MoSold', 'YrSold', 'SalePrice'],
#       dtype='object')

# ********************************************************************************
# Index(['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
#        'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
#        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
#        'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
#        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
#        'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
#        'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
#        'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
#        'SaleType', 'SaleCondition'],
#       dtype='object')




# ******************이상치 탐색, 제거************************
def detect_outliers(df, n, features):
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        
        outlier_step = 1.5 * IQR
        
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
        
    return multiple_outliers
        
Outliers_to_drop = detect_outliers(df_train, 2, ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold'])


# print(df_train.loc[Outliers_to_drop])

df_train = df_train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
# print(df_train.shape)   #(1338, 81)



# # 결측치 확인***********
# for col in df_train.columns:
#     msperc = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))
#     print(msperc)
    
    
    
# column:         Id       Percent of NaN value: 0.00%
# column: MSSubClass       Percent of NaN value: 0.00%
# column:   MSZoning       Percent of NaN value: 0.00%
# column: LotFrontage      Percent of NaN value: 17.12%
# column:    LotArea       Percent of NaN value: 0.00%
# column:     Street       Percent of NaN value: 0.00%
# column:      Alley       Percent of NaN value: 94.10%
# column:   LotShape       Percent of NaN value: 0.00%
# column: LandContour      Percent of NaN value: 0.00%
# column:  Utilities       Percent of NaN value: 0.00%
# column:  LotConfig       Percent of NaN value: 0.00%
# column:  LandSlope       Percent of NaN value: 0.00%
# column: Neighborhood     Percent of NaN value: 0.00%
# column: Condition1       Percent of NaN value: 0.00%
# column: Condition2       Percent of NaN value: 0.00%
# column:   BldgType       Percent of NaN value: 0.00%
# column: HouseStyle       Percent of NaN value: 0.00%
# column: OverallQual      Percent of NaN value: 0.00%
# column: OverallCond      Percent of NaN value: 0.00%
# column:  YearBuilt       Percent of NaN value: 0.00%
# column: YearRemodAdd     Percent of NaN value: 0.00%
# column:  RoofStyle       Percent of NaN value: 0.00%
# column:   RoofMatl       Percent of NaN value: 0.00%
# column: Exterior1st      Percent of NaN value: 0.00%
# column: Exterior2nd      Percent of NaN value: 0.00%
# column: MasVnrType       Percent of NaN value: 0.52%
# column: MasVnrArea       Percent of NaN value: 0.52%
# column:  ExterQual       Percent of NaN value: 0.00%
# column:  ExterCond       Percent of NaN value: 0.00%
# column: Foundation       Percent of NaN value: 0.00%
# column:   BsmtQual       Percent of NaN value: 2.32%
# column:   BsmtCond       Percent of NaN value: 2.32%
# column: BsmtExposure     Percent of NaN value: 2.39%
# column: BsmtFinType1     Percent of NaN value: 2.32%
# column: BsmtFinSF1       Percent of NaN value: 0.00%
# column: BsmtFinType2     Percent of NaN value: 2.39%
# column: BsmtFinSF2       Percent of NaN value: 0.00%
# column:  BsmtUnfSF       Percent of NaN value: 0.00%
# column: TotalBsmtSF      Percent of NaN value: 0.00%
# column:    Heating       Percent of NaN value: 0.00%
# column:  HeatingQC       Percent of NaN value: 0.00%
# column: CentralAir       Percent of NaN value: 0.00%
# column: Electrical       Percent of NaN value: 0.07%
# column:   1stFlrSF       Percent of NaN value: 0.00%
# column:   2ndFlrSF       Percent of NaN value: 0.00%
# column: LowQualFinSF     Percent of NaN value: 0.00%
# column:  GrLivArea       Percent of NaN value: 0.00%
# column: BsmtFullBath     Percent of NaN value: 0.00%
# column: BsmtHalfBath     Percent of NaN value: 0.00%
# column:   FullBath       Percent of NaN value: 0.00%
# column:   HalfBath       Percent of NaN value: 0.00%
# column: BedroomAbvGr     Percent of NaN value: 0.00%
# column: KitchenAbvGr     Percent of NaN value: 0.00%
# column: KitchenQual      Percent of NaN value: 0.00%
# column: TotRmsAbvGrd     Percent of NaN value: 0.00%
# column: Functional       Percent of NaN value: 0.00%
# column: Fireplaces       Percent of NaN value: 0.00%
# column: FireplaceQu      Percent of NaN value: 48.28%
# column: GarageType       Percent of NaN value: 4.86%
# column: GarageYrBlt      Percent of NaN value: 4.86%
# column: GarageFinish     Percent of NaN value: 4.86%
# column: GarageCars       Percent of NaN value: 0.00%
# column: GarageArea       Percent of NaN value: 0.00%
# column: GarageQual       Percent of NaN value: 4.86%
# column: GarageCond       Percent of NaN value: 4.86%
# column: PavedDrive       Percent of NaN value: 0.00%
# column: WoodDeckSF       Percent of NaN value: 0.00%
# column: OpenPorchSF      Percent of NaN value: 0.00%
# column: EnclosedPorch    Percent of NaN value: 0.00%
# column:  3SsnPorch       Percent of NaN value: 0.00%
# column: ScreenPorch      Percent of NaN value: 0.00%
# column:   PoolArea       Percent of NaN value: 0.00%
# column:     PoolQC       Percent of NaN value: 99.85%
# column:      Fence       Percent of NaN value: 80.94%
# column: MiscFeature      Percent of NaN value: 97.16%
# column:    MiscVal       Percent of NaN value: 0.00%
# column:     MoSold       Percent of NaN value: 0.00%
# column:     YrSold       Percent of NaN value: 0.00%
# column:   SaleType       Percent of NaN value: 0.00%
# column: SaleCondition    Percent of NaN value: 0.00%
# column:  SalePrice       Percent of NaN value: 0.00%


# for col in df_test.columns:
#     msperc = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_test[col].isnull().sum() / df_test[col].shape[0]))
#     print(msperc)
    
# column:         Id       Percent of NaN value: 0.00%
# column: MSSubClass       Percent of NaN value: 0.00%
# column:   MSZoning       Percent of NaN value: 0.27%
# column: LotFrontage      Percent of NaN value: 15.56%
# column:    LotArea       Percent of NaN value: 0.00%
# column:     Street       Percent of NaN value: 0.00%
# column:      Alley       Percent of NaN value: 92.67%
# column:   LotShape       Percent of NaN value: 0.00%
# column: LandContour      Percent of NaN value: 0.00%
# column:  Utilities       Percent of NaN value: 0.14%
# column:  LotConfig       Percent of NaN value: 0.00%
# column:  LandSlope       Percent of NaN value: 0.00%
# column: Neighborhood     Percent of NaN value: 0.00%
# column: Condition1       Percent of NaN value: 0.00%
# column: Condition2       Percent of NaN value: 0.00%
# column:   BldgType       Percent of NaN value: 0.00%
# column: HouseStyle       Percent of NaN value: 0.00%
# column: OverallQual      Percent of NaN value: 0.00%
# column: OverallCond      Percent of NaN value: 0.00%
# column:  YearBuilt       Percent of NaN value: 0.00%
# column: YearRemodAdd     Percent of NaN value: 0.00%
# column:  RoofStyle       Percent of NaN value: 0.00%
# column:   RoofMatl       Percent of NaN value: 0.00%
# column: Exterior1st      Percent of NaN value: 0.07%
# column: Exterior2nd      Percent of NaN value: 0.07%
# column: MasVnrType       Percent of NaN value: 1.10%
# column: MasVnrArea       Percent of NaN value: 1.03%
# column:  ExterQual       Percent of NaN value: 0.00%
# column:  ExterCond       Percent of NaN value: 0.00%
# column: Foundation       Percent of NaN value: 0.00%
# column:   BsmtQual       Percent of NaN value: 3.02%
# column:   BsmtCond       Percent of NaN value: 3.08%
# column: BsmtExposure     Percent of NaN value: 3.02%
# column: BsmtFinType1     Percent of NaN value: 2.88%
# column: BsmtFinSF1       Percent of NaN value: 0.07%
# column: BsmtFinType2     Percent of NaN value: 2.88%
# column: BsmtFinSF2       Percent of NaN value: 0.07%
# column:  BsmtUnfSF       Percent of NaN value: 0.07%
# column: TotalBsmtSF      Percent of NaN value: 0.07%
# column:    Heating       Percent of NaN value: 0.00%
# column:  HeatingQC       Percent of NaN value: 0.00%
# column: CentralAir       Percent of NaN value: 0.00%
# column: Electrical       Percent of NaN value: 0.00%
# column:   1stFlrSF       Percent of NaN value: 0.00%
# column:   2ndFlrSF       Percent of NaN value: 0.00%
# column: LowQualFinSF     Percent of NaN value: 0.00%
# column:  GrLivArea       Percent of NaN value: 0.00%
# column: BsmtFullBath     Percent of NaN value: 0.14%
# column: BsmtHalfBath     Percent of NaN value: 0.14%
# column:   FullBath       Percent of NaN value: 0.00%
# column:   HalfBath       Percent of NaN value: 0.00%
# column: BedroomAbvGr     Percent of NaN value: 0.00%
# column: KitchenAbvGr     Percent of NaN value: 0.00%
# column: KitchenQual      Percent of NaN value: 0.07%
# column: TotRmsAbvGrd     Percent of NaN value: 0.00%
# column: Functional       Percent of NaN value: 0.14%
# column: Fireplaces       Percent of NaN value: 0.00%
# column: FireplaceQu      Percent of NaN value: 50.03%
# column: GarageType       Percent of NaN value: 5.21%
# column: GarageYrBlt      Percent of NaN value: 5.35%
# column: GarageFinish     Percent of NaN value: 5.35%
# column: GarageCars       Percent of NaN value: 0.07%
# column: GarageArea       Percent of NaN value: 0.07%
# column: GarageQual       Percent of NaN value: 5.35%
# column: GarageCond       Percent of NaN value: 5.35%
# column: PavedDrive       Percent of NaN value: 0.00%
# column: WoodDeckSF       Percent of NaN value: 0.00%
# column: OpenPorchSF      Percent of NaN value: 0.00%
# column: EnclosedPorch    Percent of NaN value: 0.00%
# column:  3SsnPorch       Percent of NaN value: 0.00%
# column: ScreenPorch      Percent of NaN value: 0.00%
# column:   PoolArea       Percent of NaN value: 0.00%
# column:     PoolQC       Percent of NaN value: 99.79%
# column:      Fence       Percent of NaN value: 80.12%
# column: MiscFeature      Percent of NaN value: 96.50%
# column:    MiscVal       Percent of NaN value: 0.00%
# column:     MoSold       Percent of NaN value: 0.00%
# column:     YrSold       Percent of NaN value: 0.00%
# column:   SaleType       Percent of NaN value: 0.07%
# column: SaleCondition    Percent of NaN value: 0.00%



# missing = df_train.isnull().sum()
# missing = missing[missing > 0]
# missing.sort_values(inplace=True)
# missing.plot.bar(figsize = (12,6))
# plt.show()


# for col in numerical_feats:
#     print('{:15}'.format(col), 
#           'Skewness: {:05.2f}'.format(df_train[col].skew()) , 
#           '   ' ,
#           'Kurtosis: {:06.2f}'.format(df_train[col].kurt())  
#          )
    
## 수치형 변수의 Skewness(비대칭도), Kurtosis(첨도)를 확인합니다.
# 이는 분포가 얼마나 비대칭을 띄는가 알려주는 척도입니다. 
# (비대칭도: a=0이면 정규분포, a<0 이면 오른쪽으로 치우침, a>0이면 왼쪽으로 치우침)
# 비대칭도와 첨도를 띄는 변수가 여럿 보입니다.
# Target Feature인 "SalePrice" 또한 약간의 정도를 보이는 것으로 보입니다.



# corr_data = df_train[['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
#        'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
#        'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
#        'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
#        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
#        'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
#        'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
#                       'MiscVal', 'MoSold', 'YrSold', 'SalePrice']]

# colormap = plt.cm.PuBu
# sns.set(font_scale=1.0)

# f , ax = plt.subplots(figsize = (14,12))
# plt.title('Correlation of Numeric Features with Sale Price',y=1,size=18)
# sns.heatmap(corr_data.corr(),square = True, linewidths = 0.1,
#             cmap = colormap, linecolor = "white", vmax=0.8)

# plt.show()


# Heat Map은 seaborn 덕분에 직관적으로 이해가 가능하여 변수 간 상관관계에 대하여 쉽게 알 수 있습니다.
# 또한 변수 간 다중 공선성을 감지하는 데 유용합니다.
# 대각선 열을 제외한 박스 중 가장 진한 파란색을 띄는 박스가 보입니다.
# 첫 번째는 'TotalBsmtSF'와 '1stFlrSF'변수의 관계입니다.
# 두 번째는 'Garage'와 관련한 변수를 나타냅니다. 
# 두 경우 모두 변수 사이의 상관 관계가 너무 강하여 다중 공선성(MultiColarisity) 상황이 나타날 수 있습니다. 
# 변수가 거의 동일한 정보를 제공하므로 다중 공선성이 실제로 발생한다는 결론을 내릴 수 있습니다.
# 또한 확인해야할 부분은 'SalePrice'와의 상관 관계입니다. 
# 'GrLivArea', 'TotalBsmtSF'및 'OverallQual'은 큰 관계를 보입니다. 
# 나머지 변수와의 상관 관계를 자세히 알아보기 위해 Zoomed Heat Map을 확인합니다.


# sns.set()
# columns = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageCars','FullBath','YearBuilt','YearRemodAdd']
# sns.pairplot(df_train[columns],size = 2 ,kind ='scatter',diag_kind='kde')
# plt.show()

# for catg in list(categorical_feats) :
#     print(df_train[catg].value_counts())
#     print('#'*50)


num_strong_corr = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageCars',
                   'FullBath','YearBuilt','YearRemodAdd']

num_weak_corr = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1',
                 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF','LowQualFinSF', 'BsmtFullBath',
                 'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                 'Fireplaces', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF','OpenPorchSF',
                 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

catg_strong_corr = ['MSZoning', 'Neighborhood', 'Condition2', 'MasVnrType', 'ExterQual',
                    'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType']

catg_weak_corr = ['Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
                  'LandSlope', 'Condition1',  'BldgType', 'HouseStyle', 'RoofStyle', 
                  'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterCond', 'Foundation', 
                  'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 
                  'HeatingQC', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 
                  'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 
                  'SaleCondition' ]



f, ax = plt.subplots(1, 1, figsize = (10,6))
g = sns.distplot(df_train["SalePrice"], color = "b", label="Skewness: {:2f}".format(df_train["SalePrice"].skew()), ax=ax)
g = g.legend(loc = "best")

# print("Skewness: %f" % df_train["SalePrice"].skew())
# print("Kurtosis: %f" % df_train["SalePrice"].kurt())


df_train["SalePrice_Log"] = df_train["SalePrice"].map(lambda i:np.log(i) if i>0 else 0)

f, ax = plt.subplots(1, 1, figsize = (10,6))
g = sns.distplot(df_train["SalePrice_Log"], color = "b", label="Skewness: {:2f}".format(df_train["SalePrice_Log"].skew()), ax=ax)
g = g.legend(loc = "best")

# print("Skewness: %f" % df_train['SalePrice_Log'].skew())
# print("Kurtosis: %f" % df_train['SalePrice_Log'].kurt())

df_train.drop('SalePrice', axis= 1, inplace=True)


cols_fillna = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',
               'GarageQual','GarageCond','GarageFinish','GarageType', 'Electrical',
               'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st',
               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2',
               'MSZoning', 'Utilities']

for col in cols_fillna:
    df_train[col].fillna('None',inplace=True)
    df_test[col].fillna('None',inplace=True)
    
    
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print(missing_data.head(5))

df_train.fillna(df_train.mean(), inplace=True)
df_test.fillna(df_test.mean(), inplace=True)

total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print(missing_data.head(5))

# print(df_train.isnull().sum().sum(), df_test.isnull().sum().sum())


id_test = df_test['Id']

to_drop_num  = num_weak_corr
to_drop_catg = catg_weak_corr

cols_to_drop = ['Id'] + to_drop_num + to_drop_catg 

for df in [df_train, df_test]:
    df.drop(cols_to_drop, inplace= True, axis = 1)

# print(df_train.head())


# 'MSZoning'
msz_catg2 = ['RM', 'RH']
msz_catg3 = ['RL', 'FV'] 


# Neighborhood
nbhd_catg2 = ['Blmngtn', 'ClearCr', 'CollgCr', 'Crawfor', 'Gilbert', 'NWAmes', 'Somerst', 'Timber', 'Veenker']
nbhd_catg3 = ['NoRidge', 'NridgHt', 'StoneBr']

# Condition2
cond2_catg2 = ['Norm', 'RRAe']
cond2_catg3 = ['PosA', 'PosN'] 

# SaleType
SlTy_catg1 = ['Oth']
SlTy_catg3 = ['CWD']
SlTy_catg4 = ['New', 'Con']



for df in [df_train, df_test]:
    
    df['MSZ_num'] = 1  
    df.loc[(df['MSZoning'].isin(msz_catg2) ), 'MSZ_num'] = 2    
    df.loc[(df['MSZoning'].isin(msz_catg3) ), 'MSZ_num'] = 3        
    
    df['NbHd_num'] = 1       
    df.loc[(df['Neighborhood'].isin(nbhd_catg2) ), 'NbHd_num'] = 2    
    df.loc[(df['Neighborhood'].isin(nbhd_catg3) ), 'NbHd_num'] = 3    

    df['Cond2_num'] = 1       
    df.loc[(df['Condition2'].isin(cond2_catg2) ), 'Cond2_num'] = 2    
    df.loc[(df['Condition2'].isin(cond2_catg3) ), 'Cond2_num'] = 3    
    
    df['Mas_num'] = 1       
    df.loc[(df['MasVnrType'] == 'Stone' ), 'Mas_num'] = 2 
    
    df['ExtQ_num'] = 1       
    df.loc[(df['ExterQual'] == 'TA' ), 'ExtQ_num'] = 2     
    df.loc[(df['ExterQual'] == 'Gd' ), 'ExtQ_num'] = 3     
    df.loc[(df['ExterQual'] == 'Ex' ), 'ExtQ_num'] = 4     
   
    df['BsQ_num'] = 1          
    df.loc[(df['BsmtQual'] == 'Gd' ), 'BsQ_num'] = 2     
    df.loc[(df['BsmtQual'] == 'Ex' ), 'BsQ_num'] = 3     
 
    df['CA_num'] = 0          
    df.loc[(df['CentralAir'] == 'Y' ), 'CA_num'] = 1    

    df['Elc_num'] = 1       
    df.loc[(df['Electrical'] == 'SBrkr' ), 'Elc_num'] = 2 


    df['KiQ_num'] = 1       
    df.loc[(df['KitchenQual'] == 'TA' ), 'KiQ_num'] = 2     
    df.loc[(df['KitchenQual'] == 'Gd' ), 'KiQ_num'] = 3     
    df.loc[(df['KitchenQual'] == 'Ex' ), 'KiQ_num'] = 4      
    
    df['SlTy_num'] = 2       
    df.loc[(df['SaleType'].isin(SlTy_catg1) ), 'SlTy_num'] = 1  
    df.loc[(df['SaleType'].isin(SlTy_catg3) ), 'SlTy_num'] = 3  
    df.loc[(df['SaleType'].isin(SlTy_catg4) ), 'SlTy_num'] = 4 
    
new_col_HM = df_train[['SalePrice_Log', 'MSZ_num', 'NbHd_num', 'Cond2_num', 'Mas_num', 'ExtQ_num', 'BsQ_num', 'CA_num', 'Elc_num', 'KiQ_num', 'SlTy_num']]

colormap = plt.cm.PuBu
plt.figure(figsize=(10, 8))
plt.title("Correlation of New Features", y = 1.05, size = 15)
sns.heatmap(new_col_HM.corr(), linewidths = 0.1, vmax = 1.0,
           square = True, cmap = colormap, linecolor = "white", annot = True, annot_kws = {"size" : 12})
# plt.show()

df_train.drop(['MSZoning','Neighborhood' , 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType', 'Cond2_num', 'Mas_num', 'CA_num', 'Elc_num', 'SlTy_num'], axis = 1, inplace = True)
df_test.drop(['MSZoning', 'Neighborhood' , 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType', 'Cond2_num', 'Mas_num', 'CA_num', 'Elc_num', 'SlTy_num'], axis = 1, inplace = True)


print(df_train.head())

from sklearn.model_selection import train_test_split
from sklearn import metrics

X_train = df_train.drop("SalePrice_Log", axis = 1).values
target_label = df_train["SalePrice_Log"].values
X_test = df_test.values
X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size = 0.2, random_state = 2000)

import xgboost
regressor = xgboost.XGBRegressor(colsample_bytree = 0.4603, learning_rate = 0.06, min_child_weight = 1.8,
                                 max_depth= 3, subsample = 0.52, n_estimators = 2000,
                                 random_state= 7, ntrhead = -1)
regressor.fit(X_tr,y_tr)


y_hat = regressor.predict(X_tr)

# plt.scatter(y_tr, y_hat, alpha = 0.2)
# plt.xlabel('Targets (y_tr)',size=18)
# plt.ylabel('Predictions (y_hat)',size=18)
# plt.show()

regressor.score(X_tr,y_tr)

y_hat_test = regressor.predict(X_vld)


# plt.scatter(y_vld, y_hat_test, alpha=0.2)
# plt.xlabel('Targets (y_vld)',size=18)
# plt.ylabel('Predictions (y_hat_test)',size=18)
# plt.show()

regressor.score(X_vld,y_vld)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_tr, y = y_tr, cv = 10)
print(accuracies.mean())
print(accuracies.std())

use_logvals = 1

pred_xgb = regressor.predict(X_test)

sub_xgb = pd.DataFrame()
sub_xgb['Id'] = id_test
sub_xgb['SalePrice'] = pred_xgb

if use_logvals == 1:
    sub_xgb['SalePrice'] = np.exp(sub_xgb['SalePrice']) 

sub_xgb.to_csv('./_save/house_price/subtest1.csv',index=False)

# use_logvals는 Log를 취해준 Target feature을 exp해주기 위해 사용되는 스위치 역할입니다.
# 제대로 된 예측을 위해 학습 후 Log변환을 풀어줘야하기 때문입니다.
# 이 셀의 코드를 통해 submission까지 완료하게됩니다.