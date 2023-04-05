import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import imgaug.augmenters as iaa

# Load data
path = './_data/air/'
train_data = pd.read_csv(path + 'train_data.csv')
test_data = pd.read_csv(path + 'test_data.csv')

# Reshape train_data into a compatible shape
train_data_reshaped = train_data.values.reshape(-1, 7, 1)

# Define augmentations
augmentations = iaa.Sequential([
    iaa.Affine(rotate=(-180, 180)),   # Random rotation
    iaa.AdditiveGaussianNoise(scale=(0, 0.1)),  # Random noise
    iaa.Affine(scale=(0.5, 2.0))     # Random scaling
])

# Apply augmentations to train_data
train_data_augmented = augmentations.augment_images(train_data_reshaped).reshape(-1, 7)

# Concatenate original and augmented train_data
train_data_concat = pd.concat([train_data, pd.DataFrame(train_data_augmented, columns=train_data.columns)], axis=0)

# Scale data
data = pd.concat([train_data_concat, test_data], axis=0)
def type_to_HP(type):
    HP = [30, 20, 10, 50, 30, 30, 30, 30]
    gen = (HP[i] for i in type)
    return list(gen)

data['type'] = type_to_HP(data['type'])

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Train k-means model
n_clusters = 2 # assuming there are normal and anomalous data points
model = KMeans(n_clusters=n_clusters, random_state=42)
model.fit(data_scaled)

# Predict anomalies in test data
test_data_scaled = scaler.transform(test_data)
labels = model.predict(test_data_scaled)
predictions = [1 if label == 1 else 0 for label in labels]

# Save predictions to submission file
submission = pd.read_csv(path + 'answer_sample.csv')
submission['label'] = pd.DataFrame({'Prediction': predictions})
submission.to_csv('./_save/air/submission.csv', index=False)