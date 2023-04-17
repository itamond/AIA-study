import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load training data
path='./_data/air/'
save_path= './_save/air/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')
# Preprocess data
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

# Define the model

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(train_data.iloc[:, :-1])
y_train = train_data.iloc[:, -1]
X_test = scaler.transform(test_data.iloc[:, :-1])

# Model Definition
model = Sequential()


# Model Compilation
model.compile(loss='binary_crossentropy', optimizer='adam')

# Model Training
model.fit(X_train, y_train, batch_size=32, epochs=500, validation_split=0.1)

# Model Prediction
submission['label'] = (model.predict(X_test) > 0.5).astype(int)

# Save the results to a submission file
submission.to_csv(save_path+'submission.csv', index=False)