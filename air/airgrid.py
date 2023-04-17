import pandas as pd
from sklearn.preprocessing import StandardScaler,RobustScaler, MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# Load training data
path='./_data/air/'
save_path= './_save/air/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# # Preprocess data
# def type_to_HP(type):
#     HP=[30,20,10,50,30,30,30,30]
#     gen=(HP[i] for i in type)
#     return list(gen)
# train_data['type']=type_to_HP(train_data['type'])
# test_data['type']=type_to_HP(test_data['type'])

# Oversample the minority class
oversampler = RandomOverSampler(random_state=0)
X_train, y_train = oversampler.fit_resample(train_data.iloc[:, :-1], train_data.iloc[:, -1])

# Feature Scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(test_data.iloc[:, :-1])

# Model Definition
model = LocalOutlierFactor(contamination= 0.048, n_neighbors=37)

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
submission.to_csv(save_path+'submit_air'+date+ '_0.04795.csv', index=False)

