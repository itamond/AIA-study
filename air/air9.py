import pandas as pd
from sklearn.preprocessing import StandardScaler,RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
# Load training data
path='./_data/air/'
save_path= './_save/air/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')
# Preprocess data
train_data = train_data.drop(['air_end_temp','type'], axis=1)
test_data = test_data.drop(['air_end_temp','type'], axis=1)



# Preprocess data
# ...
# def type_to_HP(type):
#     HP=[30,20,10,50,30,30,30,30]
#     gen=(HP[i] for i in type)
#     return list(gen)
# train_data['type']=type_to_HP(train_data['type'])
# test_data['type']=type_to_HP(test_data['type'])

# Define the model
# Feature Scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train_data.iloc[:, :-1])
X_test = scaler.transform(test_data.iloc[:, :-1])

# Model Definition

pca = PCA(n_components=3)
X_test = pca.fit_transform(X_test)
X_train = pca.transform(X_train)

model = LocalOutlierFactor(contamination= 0.04808, n_neighbors=37,
                           metric='minkowski')
# Model Training
model.fit(X_train)

# Model Prediction
y_pred = model.fit_predict(X_test)
submission['label'] = [1 if label == -1 else 0 for label in y_pred]
print(submission.value_counts())
print(submission['label'].value_counts())
# Save the results to a submission file
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  
submission.to_csv(save_path+'submit_air'+date+ '_0.048.csv', index=False)

#neighbors 250 cont 0.04   0.8852102274
#neighbors 25 cont 0.052   0.858864335



# 네이버 37
# 0478  0.9520089276
# 0477 0.9520089276

#4807  0.9521970344   356       2207번 1
#4803  0.9531917404   355       2207번 0
#4795  0.9531917404   355





#네이버 370
# 3     1         232
# 1     1          90
# 2     1          10
# 0     1           9
# 4     1           8
# 6     1           5

