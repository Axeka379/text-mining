# text-mining
Project for text mining, 

## Analyzing Premier League Supporters Opinion on VAR

### Abstract
This work conducts a sentiment analysis on Premier League supporters opinion on VAR. Using data from twitter in the form of tweets, it is investigated if the overall opinion from the supporters is negative or positive. After the tweets have been cleaned and made available for analysis, a model is trained on tweets that have their sentiment manually annotated and are then used to classify the tweets from the Premier League supporters to find the sentiment of each tweet. It is also investigated if the league position of the team and the number of VAR decisions for each team has any impact. The results show that the overall opinion of the supporters regarding VAR is negative and that the combination of a lower league table position and VAR decisions given against a team have an impact on the supporters reaction, although it is not clear which factor has the largest impact. It is concluded that more data from a full season is needed in order to confirm the results.



## Files
The relevant code for filtering data, training models, classifying etc, can be found in the python notebook PLTweetclassifier. Most of the work done was performed using google colab. 

There also exists some python files which where used to merge datasets together into one. 

## Dataset
Dataset used can be found at: https://www.kaggle.com/wjia26/epl-teams-twitter-sentiment-dataset
