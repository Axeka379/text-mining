import pandas as pd
import bz2
import matplotlib.pyplot as plt
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

df = pd.read_csv("modified_tweets.csv")

#Global
stop_words = set(stopwords.words('english'))

def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
        u"\U00002500-\U00002BEF"  
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

def clean_tweet(text):
    text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = remove_emojis(text)
    
    text_tokens = word_tokenize(text)
    filtered_words = [w for w in text_tokens if not w in stop_words]

    return " ".join(filtered_words)




print(df.size)

'''
for index, row in df1.iterrows():
    if "VAR" in row['text'] and "RT" not in row['text'][0:2]:
        print(index, row['text'])
        #print(index, row['text']) 
'''

#print(df)
#test_df = df.groupby('partition_1').value_counts()
#test_df = df['file_name'].value_counts().plot(kind='bar', rot=0)
#plt.show()
print(df.iloc[0]['text'])
print(clean_tweet(df.iloc[0]['text']))

#for index, row in df.iterrows():
#    print(index, row['text'])

    




