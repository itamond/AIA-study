# Import required libraries
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

# Load the training data
train_df = pd.read_csv('./_data/aif/train_all.csv').drop('date',axis=1)

# Normalize the data
train_data = train_df['PM'].values.reshape(-1, 1)
# train_data = train_df
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)

# Define a function to create input/output data sequences
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Create sequences of input/output data
seq_length = 24
X, y = create_sequences(train_data, seq_length)

# Split the data into training and validation sets
train_size = int(len(X) * 0.7)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=1024)

# Load the test data
test_df = pd.read_csv('./_data/aif/test_all.csv').drop('date',axis=1)

# Normalize the test data
test_data = test_df['PM'].values.reshape(-1, 1)
test_data = scaler.transform(test_data)

# Make predictions for the next 72 hours
num_predictions = 72
predicted_values = []
input_data = test_data[:seq_length].reshape(1, seq_length, 1)
for i in range(num_predictions):
    prediction = model.predict(input_data)[0]
    predicted_values.append(prediction)
    input_data = np.append(input_data[:,1:,:], [[prediction]], axis=1)

# Inverse transform the predicted values
predicted_values = scaler.inverse_transform(np.array(predicted_values).reshape(-1,1)).flatten()

# Save the predicted values as a submit file
submit_df = pd.DataFrame({'predicted_values': predicted_values})
submit_df.to_csv('./_data/aif/submit.csv', index=False)

# Repeat the process for subsequent test data with the same structure
while True:
    test_df = pd.read_csv('./_data/aif/test.csv')
    if len(test_df) == 48:
        test_data = test_df['PM'].values.reshape(-1, 1)
        test_data = scaler.transform(test_data)
        predicted_values = []
        input_data = test_data[:seq_length].reshape(1, seq_length, 1)
        for i in range(num_predictions):
            prediction = model.predict(input_data)[0][0]
            predicted_values.append(prediction)
            input_data = np.append(input_data[:,1:,:], [[prediction]], axis=1)
        predicted_values = scaler.inverse_transform(np.array(predicted_values).reshape(-1,1)).flatten()
        submit_df = pd.DataFrame({'predicted_values': predicted_values})
        submit_df.to_csv('/_data/aif/anser_sample.csv', mode='a', header=False, index=False)
    else:
        break