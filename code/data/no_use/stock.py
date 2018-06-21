import pandas as pd
import numpy as np

df = pd.read_csv('stock.txt')
df = df.sort_values('Asset')
df['ratio'] = df.apply(lambda row: row.Current_Assets/row.Asset,axis=1)
df = df.sort_values(['ratio','EPS_2017'],ascending=False)
print(df)
