# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 20:10:52 2020

@author: Sancha Azevedo

"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def le_arquivo(name, separador=','):
   
    try:
        arq = pd.read_csv(name, sep=separador)
        return arq
    except FileNotFoundError:
        print('Error: File not found.')
        

# Opening database file
dados = le_arquivo("C:/Users/Sancha/Documents/IAeML/myCodes/diversos/desafio_indt/Tweets.csv")        
print(dados.head(10))
print(dados.columns)

# Printing class counts
print(dados.airline_sentiment.value_counts())
dados.airline_sentiment.value_counts().plot(kind='bar')

# Percentage of each class
neg_perc = len(dados.loc[dados['airline_sentiment']=='negative'])/dados.shape[0] * 100
pos_perc = len(dados.loc[dados['airline_sentiment']=='positive'])/dados.shape[0] * 100
neu_perc = len(dados.loc[dados['airline_sentiment']=='neutral'])/dados.shape[0] * 100


# Are there tweets duplicated?
base_sem_duplicados = dados.drop_duplicates(['text'])

# Are there missvalues?
print(dados.isnull().sum().sort_values())
#dados.isnull().sum().plot(kind='bar')

# looking at confidence of sentiment - taking with confidence>0.65
base = base_sem_duplicados.loc[dados['airline_sentiment_confidence']>0.65]

# Printing class counts
print(base.airline_sentiment.value_counts())
base.airline_sentiment.value_counts().plot(kind='bar')

#%%
base['Count'] = 1
base_agr = base.groupby(['airline','airline_sentiment'], as_index=False).sum()

# Ploting class by airline company
plt.figure(figsize=(30,30))
sns.catplot(x= 'airline', 
            y='Count', hue='airline_sentiment', 
            kind='bar',
            height=10.5, data=base_agr);
plt.show()


#%%
#  %NUll for negative reason by  class 
neg = base.loc[base['airline_sentiment']=='negative']
neg_sem_null = neg.dropna(subset=['negativereason'])

neg_reason = neg.groupby(['negativereason'], as_index=False).sum()


#%%% Plotting  %NUll for negative reason by  class 
sns.barplot(x="Count", y="negativereason", data=neg_reason)
plt.show()

#%%
neg_reason = neg.groupby(['airline','negativereason'], as_index=False).sum()
#%%
plt.figure(figsize=(30,30))
sns.catplot(x= 'Count', 
            y='airline', hue='negativereason', 
            kind='bar',
            height=10.5, data=neg_reason);
plt.show()

#%%

print('Textos distintos:{}'.format(len(base['text'].unique())))

#%%
# Shows quantity of tweets by user, and retweets
users = list(base['name'].unique())
print( "Users uniques:{}".format( len(users) ))

by_user = base.groupby(['name'], as_index=False).sum()
#%%
#base['airline'].value_counts().plot(kind='pie', autopct='%1.0f%%')
#plt.figure(figsize=(30,30))
#sns.catplot(x= 'Count', 
 #           y='name',  
  #          kind='bar',
   #         height=10.5, data=by_user);
#plt.show()




#%%
# Saving in a file just text and label from database
cols_names = ['text', 'airline_sentiment']
base_final = pd.DataFrame()
base_final['text'] = base['text']
base_final['airline_sentiment'] =  base['airline_sentiment']

base_final.to_csv('sentiment_class.csv', header=cols_names, index=False)