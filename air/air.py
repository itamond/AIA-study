import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load training and testing data
path='./_data/air/'

train_data = pd.read_csv(path + 'train_data.csv')
test_data = pd.read_csv(path + 'test_data.csv')

# Separate labels from features in training data
X_train = train_data.drop(columns=['label'])
y_train = train_data['label']

# Scale training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Split training data into train/validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define binary classification model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model on training data
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[es])

# Evaluate model on testing data
X_test = test_data.drop(columns=['label'])
y_test = test_data['label']
X_test = scaler.transform(X_test)
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# Calculate F1 score
f1 = f1_score(y_test, y_pred, average='macro')
print('Macro F1 score:', f1)

# Save predictions to submission.csv
submission = pd.DataFrame({'label': y_pred.reshape(-1)})
submission.to_csv('submission.csv', index=False)