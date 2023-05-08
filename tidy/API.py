# import warnings
# warnings.filterwarnings('ignore')

# Handling Data
import numpy as np
import pandas as pd
# Visialization
import matplotlib.pyplot as plt
import seaborn as sns
# For processing 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
# nltk.download('punkt')
# nltk.download('stopwords')

#스케일러
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer    #standard+MinMaxScaler

#인코딩
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

#컬런 증폭
from sklearn.preprocessing import PolynomialFeatures

from tensorflow.keras.preprocessing.image import ImageDataGenerator #이미지 전처리
from keras.preprocessing.text import Tokenizer                      #자연어(텍스트) 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_selection import SelectFromModel               #컬럼선택

# ML
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict #교차검증
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV   
from bayes_opt import BayesianOptimization     #최댓값 찾기
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.feature_extraction.text import CountVectorizer
#불균형데이터 처리(증폭)
from imblearn.over_sampling import SMOTE 
#차원축소
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #LDA
#결측치처리
from sklearn.experimental import enable_iterative_imputer #IterativeImputer : 정식버전 아님 / SimpleImputer, KNNImputer : 정식버전
from sklearn.impute import IterativeImputer #interpolation과 비슷(선형회귀선 값과 비슷 )
from sklearn.impute import SimpleImputer #돌리다, 전가하다(결측치에 대한 책임을 돌리다..) => 결측치 대체
from sklearn.impute import KNNImputer    #최근접이웃값 대체 
from impyute.imputation.cs import mice   #pd=>numpy로 변경 (mice에서는 numpy형태로 넣어주기)
#이상치 확인
from sklearn.covariance import EllipticEnvelope

#상관관계 확인
from statsmodels.stats.outliers_influence import variance_inflation_factor  #통계적 기법에서 사용


# ML model
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# DL model
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from tensorflow.python.keras.layers import MaxPooling1D, BatchNormalization, Flatten, Dropout
from tensorflow.keras.layers import Conv1D, Conv2D, LSTM, Reshape, Embedding 
from tensorflow.keras.layers import concatenate, Concatenate     #앙상블

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

#Metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

#######################################################################################
#시간저장
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M") 

#그래프
import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()



#그래프
#그림(그래프)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic' #한글깨짐 해결 #다른 폰트 필요시 윈도우 폰트파일에 추가해줘야함
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red') #marker='.' 점점으로 표시->선이됨
plt.plot(hist.history['val_loss'], marker='.', c='blue')
plt.title('따릉이')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend(('로스', '발_로스')) # 범례 표시 
plt.grid()    #격자표시 
plt.show()

#시본 API 함수로 
def show_dist_plot(df, columns):
    for column in columns:
        f, ax = plt.subplots(1,2,figsize=(16,4))
        sns.stripplot(x=df['quality'],y=df[column], ax=ax[0],hue=df['quality'])
        sns.violinplot(data=df, x='quality', y=column, ax=ax[1])
        
show_dist_plot(train_csv, numerical_columns)


plt.figure(figsize=(18,8))
corr= train_csv.corr()
sns.heatmap(corr, annot=True, square=False, vmin=-.6, vmax=1.0);
plt.show()