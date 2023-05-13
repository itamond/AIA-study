import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise import KNNWithZScore
# Importing other modules from Surprise
from surprise import Dataset
from surprise import KNNWithZScore
from surprise.model_selection import GridSearchCV, RandomizedSearchCV
import tensorflow as tf
import h2o

h2o.init()


train_data = h2o.import_file('./dacon_book/open/train.csv', index_col=0)
test_data = h2o.import_file('./dacon_book/open/test.csv', index_col=0)
submission = pd.read_csv('./dacon_book/open/sample_submission.csv', index_col=0)
x = train_data.columns
y = "Book-Rating"
x.remove(y)

from h2o.automl import H2OAutoML

aml = H2OAutoML(
    max_models=10,
    seed=44,
    max_runtime_secs=360,
    sort_metric='RMSE',
    stopping_rounds = 4

)

aml.train(
    x=x,
    y=y,
    training_frame=train_data
)

leaderboard = aml.leaderboard
print(leaderboard.head())


test = h2o.import_file('./dacon_book/open/test.csv')

model = aml.leader

pred = model.predict(test)

pred_df = pd.DataFrame(pred.as_data_frame())

print(pred_df)


submission['Book-Rating'] = pred_df['predict']

submission.to_csv('./dacon_book/subsub.csv',index=False)