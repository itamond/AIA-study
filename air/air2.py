import pandas as pd
from sklearn.ensemble import IsolationForest

# Load train and test data
path='./_data/air/'
save_path= './_save/air/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

train_data = train_data.drop(['out_pressure'],axis=1)
test_data = test_data.drop(['out_pressure'],axis=1)
# Combine train and test data
data = pd.concat([train_data, test_data], axis=0)

# Preprocess data
...
def type_to_HP(type):
    HP=[30,20,10,50,30,30,30,30]
    gen=(HP[i] for i in type)
    return list(gen)
train_data['type']=type_to_HP(train_data['type'])
test_data['type']=type_to_HP(test_data['type'])






# Train isolation forest model on train data
model = IsolationForest(n_estimators=2000,random_state=49210,max_samples='auto',
                        max_features=7, bootstrap=False,)

model.fit(train_data)

# Predict anomalies in test data
predictions = model.predict(test_data)

# Save predictions to submission file
new_predictions = [0 if x == 1 else 1 for x in predictions]
submission['label'] = pd.DataFrame({'Prediction': new_predictions})
submission.to_csv(save_path+'submission2.csv', index=False)



########################################################################


# import pandas as pd
# from sklearn.utils import shuffle
# import numpy as np

# path='./_data/air/'
# save_path= './_save/air/'
# train_data = pd.read_csv(path+'train_data.csv')
# test_data = pd.read_csv(path+'test_data.csv')
# submission = pd.read_csv(path+'answer_sample.csv')

# train_data = train_data.drop(['out_pressure'],axis=1)
# test_data = test_data.drop(['out_pressure'],axis=1)

# def type_to_HP(type):
#     HP=[30,20,10,50,30,30,30,30]
#     gen=(HP[i] for i in type)
#     return list(gen)

# train_data['type'] = type_to_HP(train_data['type'])
# test_data['type'] = type_to_HP(test_data['type'])

# times = 10 # set the number of times to amplify
# shuffle_frac = 0.0001 # set the fraction of numerical columns to shuffle

# # Amplify train data
# amplified_data = [train_data]
# for i in range(times - 1):
#     shuffled_data = train_data.copy()
#     # shuffle only a fraction of the numerical columns
#     numeric_cols = shuffled_data.select_dtypes(include=np.number).columns
#     num_shuffle_cols = int(np.ceil(shuffle_frac * len(numeric_cols)))
#     shuffle_cols = np.random.choice(numeric_cols, num_shuffle_cols, replace=False)
#     shuffled_data[shuffle_cols] = shuffle(shuffled_data[shuffle_cols])
#     amplified_data.append(shuffled_data)

# # Concatenate amplified data and save to file
# amplified_data = pd.concat(amplified_data, axis=0)
# amplified_data.to_csv(path+'amplified_train_data.csv', index=False)