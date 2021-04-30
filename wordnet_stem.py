#!/usr/bin/env python
# coding: utf-8

# In[8]:


import string
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')


# In[9]:


import pandas as pd
import orchest
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[10]:


df = orchest.get_inputs()["stop_dataframe"]


# In[11]:


from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
# Use English stemmer.
lemmatizer = WordNetLemmatizer()
#stemmer = SnowballStemmer("english")
df['reviews_stemmed'] = df['reviews_without_stop'].apply(lambda y:' '.join([lemmatizer.lemmatize(y) for y in y.split()]))
#df['reviews_stemmed'] = df['reviews_without_stop'].apply(lambda y:' '.join([stemmer.stem(y) for y in y.split()]))


# In[12]:


# DEFINE FUNCTION TO CALCULATE SENTIMENT SCORE 
def sentimentScore(sentences):
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for sentence in sentences:
        vs = analyzer.polarity_scores(sentence)
        #print(str(vs))
        results.append(vs) 
    return results
sentiment = sentimentScore(df['reviews_stemmed'])


# In[6]:


sentiment 


# In[13]:


sentiment_df = pd.DataFrame(sentiment)
df.index = sentiment_df.index
sentiment_add_df = pd.concat([df, sentiment_df], axis=1)
sentiment_add_df.head(100)


# In[8]:


orchest.output(sentiment_add_df, name="sentiment2_dataframe")


# In[14]:


df['reviews_stemmed']


# In[ ]:




