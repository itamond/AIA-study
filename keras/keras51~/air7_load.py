import pandas as pd
import joblib
import numpy as np

# Load test data
path='./_data/air/'
save_path= './_save/air/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')
# Preprocess data
# ...

# Remove the 'type' column since it is not used in clustering
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

# Load the trained K-means model
kmeans = joblib.load(save_path+'kmeans_model.pkl')

# Predict the clusters for each data point in the test data
test_clusters = kmeans.predict(test_data)

# Calculate the distance between each test data point and the centroid of its assigned cluster
distances = []
for i in range(len(test_data)):
    centroid = kmeans.cluster_centers_[test_clusters[i]]
    distance = np.linalg.norm(test_data.iloc[i] - centroid)
    distances.append(distance)

# Set a threshold for the distance beyond which a data point is considered abnormal
threshold = 100

# Label each test data point as normal (0) or abnormal (1) based on the distance from its cluster centroid
test_data['label'] = np.where(np.array(distances) > threshold, 1, 0)

# Save the results to a submission file
submission['label'] = test_data[['label']]
submission.to_csv(save_path+'submission.csv', index=False)