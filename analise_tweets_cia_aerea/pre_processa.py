# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 03:02:42 2020

@author: Sancha Azevedo


"""

from nltk.tokenize import TweetTokenizer
from string import punctuation 
from nltk.corpus import stopwords 
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import nltk
nltk.download('stopwords')
nltk.download('punkt')
#%%

# Text Pre-processing
def Preprocessing(texto):
    # convert text to lower
    texto = texto.lower()
    
    #remove URL
    texto = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', texto)
    
    # remove usernames
    texto = re.sub('@[^\s]+', '', texto)
    
    # remove the # simbol in #hashtag
    texto = re.sub(r'#([^\s]+)', r'\1', texto) 
       
    #remove stopwords and punctuation
    stopwords_list = set(stopwords.words('english') + list(punctuation) )
    palavras = [i for i in texto.split() if not i in stopwords_list]
    return (" ".join(palavras))


#%%

def input_treat(name='C:/Users/Sancha/Documents/IAeML/myCodes/diversos/desafio_indt/sentiment_class.csv', separador=','):
    # Reading text-label file
    base = pd.read_csv(name, sep=separador)
    
    # Aparting: Text and Label 
    textos = base['text']
    # mapping classes: neg:0, pos:1, neu:2
    classes = base['airline_sentiment'].map({'negative':0,
                                             'positive':1,
                                             'neutral':2})
    
    # Pre processing each tweet from base
    tweets = [Preprocessing(tweet) for tweet in textos]
    
    return tweets, classes

#%%
    
def vetoriza(tweets, ngram_min=1,ngram_max=1):
    
    # Vetctorize data from text, using tweet tokenizer
    tweet_tokenizer = TweetTokenizer() 
    
    vetorizador = CountVectorizer(analyzer="word", 
                                  ngram_range = (ngram_min, ngram_max),
                                  tokenizer=tweet_tokenizer.tokenize)
    tweets_freq = vetorizador.fit_transform(tweets)
    print(tweets_freq.shape)
     
    return tweets_freq

