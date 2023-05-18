import pandas as pd
import numpy as np
import random
import os
import optuna

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(91345)  # Fix seed

path = './dacon_crime/'
save_path = './dacon_crime/'

train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

# Feature Engineering
train['날씨'] = train['강수량(mm)'] + train['강설량(mm)'] + train['적설량(cm)']
test['날씨'] = test['강수량(mm)'] + test['강설량(mm)'] + test['적설량(cm)']

x_train = train.drop(['ID', 'TARGET'], axis=1)  # Removed '강수량(mm)', '적설량(cm)'
y_train = train['TARGET']
x_test = test.drop('ID', axis=1)

le = LabelEncoder()

# Label Encoding for '요일' and '범죄발생지' features
for feature in ['요일', '범죄발생지']:
    x_train[feature] = le.fit_transform(x_train[feature])
    x_test[feature] = le.transform(x_test[feature])

ordinal_features = ['요일', '범죄발생지']

# One-hot encoding for categorical features
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
x_train_ohe = ohe.fit_transform(x_train[ordinal_features])
x_test_ohe = ohe.transform(x_test[ordinal_features])

x_train = pd.concat([x_train, pd.DataFrame(x_train_ohe, columns=ohe.get_feature_names_out(ordinal_features))], axis=1)
x_test = pd.concat([x_test, pd.DataFrame(x_test_ohe, columns=ohe.get_feature_names_out(ordinal_features))], axis=1)

# Scaling the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Handling imbalanced data
smote = SMOTE(random_state=10, k_neighbors=10)
x_train, y_train = smote.fit_resample(x_train, y_train)

# Define the objective function for Optuna
def objective(trial):
    params_xgb = {
        'max_depth': trial.suggest_categorical('max_depth', [6, 12]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.25, 0.5]),
        'n_estimators': trial.suggest_categorical('n_estimators', [5, 11]),
        'min_child_weight': trial.suggest_categorical('min_child_weight', [1, 3]),
        'subsample': trial.suggest_categorical('subsample', [0.6, 0.9]),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.6, 0.8]),
        'max_bin': trial.suggest_categorical('max_bin', [10, 20]),
        'reg_lambda': trial.suggest_categorical('reg_lambda', [1, 5]),
        'reg_alpha': trial.suggest_categorical('reg_alpha', [0.01, 0.1]),
    }

    xgb_model = XGBClassifier(
        random_state=10,
        use_label_encoder=False,
        tree_method='gpu_hist',
        gpu_id=0,
        objective='binary:logistic',
        **params_xgb
    )

    # Split the data into train and validation sets for early stopping
    x_train_split, x_val, y_train_split, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=10)

    # Fit the model and track evaluation metric
    xgb_model.fit(
        x_train_split,
        y_train_split,
        early_stopping_rounds=10,
        eval_set=[(x_val, y_val)],
        eval_metric='logloss',
        verbose=False
    )

    # Retrieve the best evaluation metric score
    best_score = xgb_model.best_score

    return best_score


# Run Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Get the best parameters and score
best_params = study.best_params
best_score = study.best_value

# Print the best parameters and score
print('XGBoost Best Parameters:', best_params)
print('XGBoost Best Score:', best_score)

# Get the best model
best_model = XGBClassifier(
    random_state=10,
    use_label_encoder=False,
    tree_method='gpu_hist',
    gpu_id=0,
    objective='binary:logistic',
    **best_params
)

# Fit the model with the full dataset
best_model.fit(x_train, y_train)

# Predict
pred = best_model.predict(x_test)

# Read the submission file
submit = pd.read_csv(path + 'sample_submission.csv')

# Assign the predicted values to the TARGET column
submit['TARGET'] = pred

# Save the predicted results to a file
submit.to_csv(save_path + 'submit.csv', index=False)