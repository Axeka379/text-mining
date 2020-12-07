import pandas as pd
import bz2

df = pd.read_csv("modified_tweets.csv")
       
print(df.size)

'''
for index, row in df1.iterrows():
    if "VAR" in row['text'] and "RT" not in row['text'][0:2]:
        print(index, row['text'])
        #print(index, row['text']) 
'''

print(df)
#test_df = df.groupby('partition_1').value_counts()
test_df = df['file_name'].value_counts()

print(test_df)


#print(result.size)

