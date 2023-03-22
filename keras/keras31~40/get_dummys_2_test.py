import pandas as pd


#1. 

x = ['바나나', 1, '사과', 123]

x = pd.get_dummies(x)

print(x)