import pandas as pd
import bz2
import matplotlib.pyplot as plt

df1 = pd.read_csv("2020-07-09_from_2020-09-19.csv")
df2 = pd.read_csv("2020-09-20_from_2020-10-13.csv")
       
frames = [df1, df2]

result_df = pd.concat(frames)

print(result_df['file_name'].value_counts())

#Regex out?
#print("First dataset", result_df.size)
new_df = result_df[result_df['text'].str.contains(r'\bVAR\b|\bvar\b|\bVar\b|\bvideo assistant referee\b')]

print(new_df['file_name'].value_counts())


#print("Only the ones mentioning VAR", new_df.size)
new_df = new_df[~new_df['text'].str.startswith('RT')]

print(new_df['file_name'].value_counts())
print(new_df.shape)
'''

print("No RT", new_df.size)
new_df.to_csv('modified_tweets.csv', index=False)
'''