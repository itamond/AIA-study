import pandas as pd
import numpy as np
from scipy.stats import norm

# Load train and test data
path='./_data/air/'
save_path= './_save/air/'
train_data = pd.read_csv(path+'amplified_train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# Combine train and test data
data = pd.concat([train_data, test_data], axis=0)
# test_data = test_data.drop(['out_pressure'],axis=1)

# Preprocess data
# ...

# Train Gaussian distribution model on train data
mu = data.mean(axis=0)
sigma = data.std(axis=0)

# Predict anomalies in test data
threshold = 0.05  # set threshold for anomaly detection
predictions = np.array([1 if np.sum(norm.logpdf(test_data.values[i,:], mu, sigma)) < threshold else 0 for i in range(test_data.shape[0])])

# Save predictions to submission file
submission['label'] = pd.DataFrame({'Prediction': predictions})
submission.to_csv(save_path+'submission.csv', index=False)