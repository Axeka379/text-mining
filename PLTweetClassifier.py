import pandas as pd
import bz2

df1 = pd.read_csv("2020-07-09_from_2020-09-19.csv")
df2 = pd.read_csv("2020-09-20_from_2020-10-13.csv")
       
print(df1.size)
print(df2.size)


