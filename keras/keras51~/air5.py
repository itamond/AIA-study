import pandas as pd
import numpy as np


submission=pd.read_csv('C:/AIA-study/_save/air/submission.csv')

print(np.unique(submission['label'], return_counts=True))