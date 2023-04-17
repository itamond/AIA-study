import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score

# Load training data
path = './_data/air/'
save_path = './_save/air/'
train_data = pd.read_csv(path + 'train_data.csv')
test_data = pd.read_csv(path + 'test_data.csv')
submission = pd.read_csv(path + 'answer_sample.csv')

# Preprocess data
def type_to_HP(type):
    HP = [30, 20, 10, 50, 30, 30, 30, 30]
    gen = (HP[i] for i in type)
    return list(gen)

train_data['hp'] = type_to_HP(train_data['type'])
test_data['hp'] = type_to_HP(test_data['type'])

# Define the models
models = [
    ('rf', RandomForestClassifier(n_estimators=5, random_state=42)),
    # ('lr', RandomForestClassifier(n_estimators=100,random_state=42)),
    ('gnb', LocalOutlierFactor(contamination= 0.048, n_neighbors=37, novelty=True))
]

# Define the blending function
def blend_predictions(predictions):
    return [1 if sum(p) >= (len(models) / 2) else 0 for p in zip(*predictions)]

# Divide the training set into subsets
X_train1, X_train2, y_train1, y_train2 = train_test_split(train_data.iloc[:, :-1], train_data.iloc[:, -1], test_size=0.001, random_state=42)

# Feature Scaling
scaler = MinMaxScaler()
X_train1 = scaler.fit_transform(X_train1)
X_train2 = scaler.transform(X_train2)
X_test = scaler.transform(test_data.iloc[:, :-1])

# Train the models on the subsets
predictions = []
for name, model in models:
    model.fit(X_train1, y_train1)
    y_pred = model.predict(X_train2)
    predictions.append(model.predict(X_test))

# Combine the predictions using blending
y_pred = blend_predictions(predictions)
submission['label'] = y_pred
submission['label'] = submission['label'].apply(lambda x: 0 if x == 1 else 1)
print(submission['label'].value_counts())

# Save the results to a submission file
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
submission.to_csv(save_path + 'submit_air' + date + '_blending.csv', index=False)
