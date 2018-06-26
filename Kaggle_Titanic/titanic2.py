import pandas as pd

df = pd.read_csv('train.csv')
df.sort_index()
#print(df)
#print (df['Age'].sum())

df = df['Age'].map(lambda a: a**2)
print(df)
print(df)
