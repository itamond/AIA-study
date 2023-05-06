import pandas as pd

# Read in the Monday1.csv file
df = pd.read_csv('./_data/dacon_air/Monday2.csv', index_col=0)

# Swap the values of the Not_Delayed and Delayed columns
df[['Not_Delayed', 'Delayed']] = df[['Delayed', 'Not_Delayed']].values

# Write the updated dataframe to a new CSV file
df.to_csv('./_data/dacon_air/day2.csv')