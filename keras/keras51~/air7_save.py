import pandas as pd
from sklearn.cluster import KMeans

# Load training data
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


# Apply K-means clustering to the data
kmeans = KMeans(n_clusters=8, init="random", n_init=10, max_iter=300)
kmeans.fit(train_data)

# Predict the clusters for each data point in the training data
train_clusters = kmeans.predict(train_data)

# Save the cluster assignments as a new column in the training data
train_data['cluster'] = train_clusters

# Save the trained K-means model for later use
import joblib
joblib.dump(kmeans, save_path+'kmeans_model.pkl')