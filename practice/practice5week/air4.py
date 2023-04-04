import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load training data
path='./_data/air/'
save_path= './_save/air/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')
# Preprocess data
# ...
# Preprocess data
# ...
train_data = train_data.drop(['out_pressure'],axis=1)
test_data = test_data.drop(['out_pressure'],axis=1)
# Combine train and test data
# data = pd.concat([train_data, test_data], axis=0)

# Preprocess data
# ...
def type_to_HP(type):
    HP=[30,20,10,50,30,30,30,30]
    gen=(HP[i] for i in type)
    return list(gen)
train_data['type']=type_to_HP(train_data['type'])
test_data['type']=type_to_HP(test_data['type'])


# K-means clustering to detect anomalies
kmeans = KMeans(n_clusters=16)
kmeans.fit(train_data)
train_data['cluster'] = kmeans.predict(train_data)

# Calculate difference between air intake flow rate and motor speed
train_data['flow_speed_diff'] = train_data['air_inflow'] - train_data['motor_rpm']

# Ensemble learning using logistic regression and neural network models
lr_model = LogisticRegression()
nn_model = MLPClassifier(hidden_layer_sizes=(100, 50))
X = train_data.drop(['cluster'], axis=1)
y = train_data['cluster']
lr_model.fit(X, y)
nn_model.fit(X, y)

# Load test data

# Apply K-means clustering to test data
test_data['cluster'] = kmeans.predict(test_data)

# Calculate difference between air intake flow rate and motor speed in test data
test_data['flow_speed_diff'] = test_data['air_inflow'] - test_data['motor_rpm']

# Use ensemble learning to predict anomalies in test data
X_test = test_data.drop(['cluster'], axis=1)
lr_pred = lr_model.predict(X_test)
nn_pred = nn_model.predict(X_test)
ensemble_pred = np.logical_or(lr_pred, nn_pred).astype(int)
test_data['label'] = ensemble_pred

# Save results to submission file
submission = test_data[['label']]
submission['label'] = submission['label'].apply(lambda x: 1 if x==0 else 0)
submission.to_csv(save_path+'submission.csv', index=False)