import pandas as pd
import numpy as np

submission = pd.read_csv('./dacon_book/open/book_3.2713.csv', index_col=0)
sub = pd.read_csv('./dacon_book/open/sample_submission.csv')

# submission = submission.drop(submission[0], axis = 1)

submission['Book-Rating'] = np.round(submission['Book-Rating'])
print(submission['Book-Rating'].shape)
print(submission)
print(sub)
sub['Book-Rating']=submission['Book-Rating']


sub.to_csv('./dacon_book/submission4.csv', index=False)