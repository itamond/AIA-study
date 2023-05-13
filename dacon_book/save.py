import pandas as pd
import numpy as np

submission = pd.read_csv('./dacon_book/open/book_0513_1221.csv', index_col=0)
sub = pd.read_csv('./dacon_book/open/sample_submission.csv')

# submission = submission.drop(submission[0], axis = 1)

submission['Book-Rating'] = np.round(submission['Book-Rating'])
print(submission['Book-Rating'].shape)
print(submission)
print(sub)
sub['Book-Rating']=submission['Book-Rating']


sub.to_csv('./dacon_book/submission2.csv', index=False)