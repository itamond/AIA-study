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
os.listdir("./_data/kaggle_house/")


df_train = pd.read_csv('./_data/kaggle_house/train.csv')
df_test = pd.read_csv('./_data/kaggle_house/test.csv')


numerical_feats = df_train.select_dtypes(include=['int64', 'float64']).to_string
categorical_feats = df_train.select_dtypes(include=['object']).to_string

print(numerical_feats)
print(categorical_feats)