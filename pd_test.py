import pandas as pd

# Sample DataFrame
data = {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}
df = pd.read_csv("test-data.csv", header=None, names=None, index_col=False)

df_2 = df.drop(columns=[df.columns[3]])
df_3_t1 = df_2.drop(columns=[df_2.columns[4]])
df_3_t2 = df_2.iloc[:, 0:4:]
df_3_t3 = df_2.iloc[:, df_2.columns != 4]


print("xd")
