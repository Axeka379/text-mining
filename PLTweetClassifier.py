import pandas as pd
import bz2
import matplotlib.pyplot as plt
import re
import string
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from bs4 import BeautifulSoup 
from html import unescape



train_df = pd.read_csv("sts_gold_tweet.csv", sep=";")
df = pd.read_csv("modified_tweets.csv")

train_df = train_df.filter(['polarity', 'tweet']).rename(columns={"tweet":"text"})
df = df.filter(['file_name', 'text'])



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
    soup = BeautifulSoup(text, 'lxml')
    text = soup.get_text() 
    text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = remove_emojis(text)
    
    text_tokens = word_tokenize(text)
    filtered_words = [w for w in text_tokens if not w in stop_words]

    return " ".join(filtered_words)

def get_feature_vector(data_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(data_fit)
    print("in get feature")
    print(len(vector.idf_))
    return vector   


def train_classifier(the_set):

    #
    #Getting a training and test set for the STS_tweets, just for measurement.

   # the_set.text = the_set['text'].apply(clean_tweet)

        tf_vector = get_feature_vector(np.array(the_set.iloc[:, 1]).ravel())
        the_set.text = the_set["text"].apply(clean_tweet)
        test_feature = tf_vector.transform(np.array(the_set.iloc[:, 1]).ravel())


'''
    x = tf_vector.transform(np.array(the_set.iloc[:, 1]).ravel())
    y = np.array(the_set.iloc[:, 1]).ravel()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)
'''
    #Naive bayes
'''
    nb_clf = MultinomialNB()
    nb_clf.fit(x_train, y_train)
    predict_nb = nb_clf.predict(x_test)
    print(accuracy_score(y_test, predict_nb))
'''

        

    #print(grid_search.best_score_)


#small code snippet to convert the sentiment values to text form. Might be useful later
def int_to_string(sentiment): 
    if(sentiment == 0):
        return "Negative"
    else:
        return "Positive"



NB_model = train_classifier(train_df)


#test_df = df.groupby('partition_1').value_counts()
#test_df = df['file_name'].value_counts().plot(kind='bar', rot=0)
#plt.show()


#print(df.iloc[0]['text'])
#print(clean_tweet(df.iloc[0]['text']))


#print(clean_tweet(str(text)))

#for index, row in df.iterrows():
#    print(index, row['text'])

