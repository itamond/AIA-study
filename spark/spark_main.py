import pandas as pd
from sklearn.linear_model import LinearRegression

# Load train data
train_df = pd.read_csv('./_data/aif/train_all.csv').drop('date',axis=1)
submission = pd.read_csv('./_data/aif/answer_sample.csv')
# Split features and target
X_train = train_df.fillna(value=1)
y_train = train_df['PM']

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Load test data
test_df = pd.read_csv('./_data/aif/test_all.csv').drop('date',axis=1)

# Make predictions
X_test = test_df.fillna(value=1)
y_pred = model.predict(X_test)

# Save predictions to submit file
pm2 = pd.DataFrame({'PM': y_pred})
submission['PM2.5'] = pm2
submission.to_csv('./_data/aif/submission1.csv')