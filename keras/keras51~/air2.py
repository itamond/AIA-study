import pandas as pd
from sklearn.ensemble import IsolationForest

# Load train and test data
path='./_data/air/'
save_path= './_save/air/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# Combine train and test data
data = pd.concat([train_data, test_data], axis=0)

# Preprocess data
# ...

# Train isolation forest model on train data
model = IsolationForest(n_estimators=3000,random_state=3245,max_samples=2463,
                        max_features=8, bootstrap=False,)

model.fit(train_data)

# Predict anomalies in test data
predictions = model.predict(test_data)

# Save predictions to submission file
new_predictions = [0 if x == 1 else 1 for x in predictions]
submission['label'] = pd.DataFrame({'Prediction': new_predictions})
submission.to_csv(save_path+'submission.csv', index=False)



