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
from xgboost import XGBClassifier


# Load train and test data
path='./_data/air/'
save_path= './_save/air/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

###############################################
# import matplotlib.pyplot as plt
# import seaborn as sns

# #corr 프린트 가능
# print(test_data.corr())
# plt.figure(figsize=(10,8))
# sns.set(font_scale=1.2)
# sns.heatmap(train_data.corr(), square=True, annot=True, cbar=True)
# plt.show()



###############################################





# 
# def type_to_HP(type):
#     HP=[30,20,10,50,30,30,30,30]
#     gen=(HP[i] for i in type)
#     return list(gen)
# train_data['type']=type_to_HP(train_data['type'])
# test_data['type']=type_to_HP(test_data['type'])
# print(train_data.columns)

# 
features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']
# features = ['out_pressure', 'motor_current', 'motor_temp', 'motor_vibe']

# Prepare train and test data
X = train_data[features]
print(X.shape)
pca = PCA(n_components=3)
X = pca.fit_transform(X)
print(X.shape)

# 
X_train, X_val = train_test_split(X, test_size= 0.9, random_state= 337)
print(X_train.shape, X_val.shape)

# #
# pca = PCA(n_components=2)
# X_train = pca.fit_transform(X_train)
# X_val = pca.fit_transform(X_val)

# 
scaler = MinMaxScaler()
train_data_normalized = scaler.fit_transform(train_data.iloc[:, :-1])
test_data_normalized = scaler.transform(test_data.iloc[:, :-1])

# 
n_neighbors = 46
contamination = 0.046111
lof = LocalOutlierFactor(n_neighbors=n_neighbors,
                         contamination=contamination,
                         leaf_size=99,
                         algorithm='auto',
                         metric='chebyshev',
                         metric_params= None,
                         novelty=False,
                         p=300
                         )
y_pred_train_tuned = lof.fit_predict(X_val)

# 
test_data_lof = scaler.fit_transform(test_data[features])
y_pred_test_lof = lof.fit_predict(test_data_lof)
lof_predictions = [1 if x == -1 else 0 for x in y_pred_test_lof]
#lof_predictions = [0 if x == -1 else 0 for x in y_pred_test_lof]


train_data['label']=np.zeros((train_data.shape[0]),np.int64)
train_data = pd.DataFrame(train_data)

test_data['label'] = pd.DataFrame({'Prediction': lof_predictions})

print(train_data.shape, test_data.shape)


data = pd.concat([train_data,test_data], axis=0)


print(data.shape)
features2 = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']
# features2 = ['out_pressure', 'motor_current', 'motor_temp', 'motor_vibe']
x = data[features2]
y = data['label']
test_data = test_data[features2]
# print(x.shape, y.shape)

model = XGBClassifier(n_estimators=1000,
                      max_depth=35,
                      learning_rate=0.001,
                      subsample=0.99,
                      colsample_bytree=0.99,
                      objective='binary:logistic',
                      reg_lambda=1,
                    #   gamma=2,
                      )

model.fit(x, y)
result = model.score(x,y)
y_predict = model.predict(test_data)

# x2_train,x2_test,y2_train,y2_test = train_test_split(x, y, 
#                                                      train_size=0.8, 
#                                                      random_state=338, 
#                                                      shuffle=True,
#                                                      stratify=y)


# #2-2 모델구성

# model = Sequential()
# model.add(Dense(64, input_dim=4))
# model.add(Dense(32, activation='selu'))
# model.add(Dense(64, activation='selu'))
# model.add(Dense(128, activation='selu'))
# model.add(Dense(64, activation='selu'))
# model.add(Dense(32, activation='selu'))
# model.add(Dense(1))

# model.compile(loss = 'mse', optimizer = 'adam')

# vl = ReduceLROnPlateau(monitor='val_loss' ,
#                        factor = 0.2,
#                        patience = 5)
# es = EarlyStopping(monitor='val_loss', 
#                    patience=30, 
#                    restore_best_weights=True,
#                    )

# model.fit(x2_train, y2_train, epochs=300, verbose=1, validation_split=0.2,
#           batch_size=16,
#           callbacks=[vl,es])

# # test_data= test_data.drop(['label'],axis=1)

# test_preds = model.predict(test_data)

# errors = np.mean(np.power(test_data - test_preds, 2), axis=1)
# y_pred = np.where(errors >=np.percentile(errors, 95), 1, 0)

import xgboost as xgb
import matplotlib.pyplot as plt
# xgb.plot_importance(model)
# xgb.plot_tree(model, num_trees=1, rankdir='LR')
# # xgb.plot_tree(model, num_trees=1, rankdir='LR', ax=2)
# fig = plt.gcf()
# fig.set_size_inches(150, 100)
# plt.show()

submission['label'] = y_predict
submission.to_csv(save_path  + '_XGB_Lambda5.csv', index=False)


