import pandas as pd
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
# Load training data
path='./_data/air/'
save_path= './_save/air/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')
# Preprocess data
train_data = train_data.drop(['out_pressure'],axis=1)
test_data = test_data.drop(['out_pressure'],axis=1)

# Preprocess data
# ...
def type_to_HP(type):
    HP=[30,20,10,50,30,30,30,30]
    gen=(HP[i] for i in type)
    return list(gen)
train_data['type']=type_to_HP(train_data['type'])
test_data['type']=type_to_HP(test_data['type'])

# Define the model
# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(train_data.iloc[:, :-1])
X_test = scaler.transform(test_data.iloc[:, :-1])

# Model Definition
model = LocalOutlierFactor(n_neighbors=20, contamination=0.04)

# Model Training
model.fit(X_train)

# Model Prediction
y_pred = model.fit_predict(X_test)
submission['label'] = [1 if label == -1 else 0 for label in y_pred]
print(submission.value_counts())
# Save the results to a submission file
submission.to_csv(save_path+'submission10.csv', index=False)