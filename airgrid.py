import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor

# Load data
path = './_data/air/'
save_path = './_save/air/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# Preprocess data
train_data = train_data.drop(['out_pressure'], axis=1)
test_data = test_data.drop(['out_pressure'], axis=1)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(train_data.iloc[:, :-1])
X_test = scaler.transform(test_data.iloc[:, :-1])

# Apply PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Define the model
model = LocalOutlierFactor(contamination=0.05)

# Fit the model and predict outliers
y_pred = model.fit_predict(X_train_pca)
y_pred_test = model.fit_predict(X_test_pca)

# Convert the prediction to binary labels (0: normal, 1: outlier)
submission['label'] = [1 if label == -1 else 0 for label in y_pred_test]
print(submission.value_counts())
print(submission['label'].value_counts())
# Save the results to a submission file
submission.to_csv(save_path+'submission_pca.csv', index=False)