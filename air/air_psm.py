import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load data
path = './_data/air/'
save_path = './_save/air/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# Preprocess data
def type_to_HP(type):
    HP=[30,20,10,50,30,30,30,30]
    gen=(HP[i] for i in type)
    return list(gen)
train_data['type']=type_to_HP(train_data['type'])
test_data['type']=type_to_HP(test_data['type'])

# Select subset of features
features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']

# Prepare train and validation data
X = train_data[features]
X_train, X_val = train_test_split(X, train_size=0.9, random_state=5555)

# Normalize data
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)
test_data_norm = scaler.transform(test_data[features])

# Define model architecture
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=X_train_norm.shape[1]))
model.add(Dense(16, activation='relu'))
model.add(Dense(X_train_norm.shape[1], activation='linear'))

# Compile model
model.compile(loss='mse', optimizer='adam')

# Define early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=20)

# Train model
history = model.fit(X_train_norm, X_train_norm, epochs=5000, batch_size=32, validation_data=(X_val_norm, X_val_norm), callbacks=[early_stop])

# Predict anomalies on test data
test_preds = model.predict(test_data_norm)
errors = np.mean(np.power(test_data_norm - test_preds, 2), axis=1)
y_pred = np.where(errors >= np.percentile(errors, 95.2), 1, 0)

# Save submission file
submission['label'] = pd.DataFrame({'Prediction': y_pred})
date = datetime.datetime.now().strftime("%m%d_%H%M%S")
submission.to_csv(save_path + 'air_' + date + '.csv', index=False)