import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


scaler = StandardScaler()
all_data = scaler.fit_transform(all_data)

pca = PCA(n_components=2, random_state=0)
all_tf = pca.fit_transform(all_data)
# print(all_tf)

# from sklearn.metrics import silhouette_samples, silhouette_score

# sc_max = 0
# for i in range(3, 13):
#     for j in range(3, 13):    
#         dbscan3 = DBSCAN(eps=0.1 * i, min_samples=j, metric='euclidean')
#         aaa = dbscan3.fit_predict(all_tf)
        
#         sc = silhouette_score(all_tf, aaa)
#         print("실루엣 스코어 : ", sc)
        
#         print('스캔 : ', i, j, ":", 
#             np.unique(dbscan3.labels_, return_counts=True)) 

#         if sc > sc_max:
#             sc_max = sc
#             best_parameters = {'eps' : i, 'min_samples' : j}

# print("최고 실루엣 : ", sc_max)
# print("최적의 매개변수 : ", best_parameters)
# 최고 실루엣 :  0.7814148871578233
# 최적의 매개변수 :  {'eps': 7, 'min_samples': 3}


dbscan5 = DBSCAN(eps= 0.7, min_samples= 3, metric='euclidean')
bbb = dbscan5.fit_predict(all_tf)
print(np.unique(bbb, return_counts=True))       # (array([0, 1], dtype=int64), array([9836,   16], dtype=int64))
print(np.unique(bbb[2463:], return_counts=True))
print(bbb.shape)