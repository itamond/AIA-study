

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.datasets import load_digits, load_wine, fetch_covtype
from tensorflow.keras.datasets import cifar100
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 1. 데이터
# x, y = load_iris(return_X_y=True)
# x, y = load_breast_cancer(return_X_y=True)
# x, y = load_digits(return_X_y=True)
# x, y = load_wine(return_X_y=True)
# x, y = fetch_covtype(return_X_y=True)

datasets = [load_iris,
            load_breast_cancer,
            load_digits,
            load_wine,
            # fetch_covtype
            ]

dataname = ['아이리스',
            '캔서',
            '디짓스',
            '와인',
            # '코브타입'
            ]


lda = LinearDiscriminantAnalysis()

print('========================================')
for i, v in enumerate(datasets):
    x, y = v(return_X_y=True)
    x_lda = lda.fit_transform(x, y)
    print(dataname[i], ':', x.shape, '->', x_lda.shape)
    lda_EVR=lda.explained_variance_ratio_
    cumsum = np.cumsum(lda_EVR)
    print(cumsum)
    print('========================================')


# ========================================
# 아이리스 : (150, 4) -> (150, 2)
# [0.9912126 1.       ]
# ========================================
# 캔서 : (569, 30) -> (569, 1)
# [1.]
# ========================================
# 디짓스 : (1797, 64) -> (1797, 9)
# [0.28912041 0.47174829 0.64137175 0.75807724 0.84108978 0.90674662
#  0.94984789 0.9791736  1.        ]
# ========================================
# 와인 : (178, 13) -> (178, 2)
# [0.68747889 1.        ]
# ========================================