# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 21:03:06 2020

@author: Sancha Azevedo


"""
#%%

from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pre_processa as pp
from sklearn import svm
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE

import numpy as np
#from sklearn.naive_bayes import MultinomialNB


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
#from sklearn.metrics import balanced_accuracy_score

#%%
# Splits data into traning (70%) and test (30%) sets
def train_test_sets(base, classes):
    
    X_train,X_test,y_train,y_test = train_test_split(base, classes,
                                                     test_size = 0.30, 
                                                     random_state = 42
                                                     )
    # Show the results of the split
    print("Training set has {} samples.".format(X_train.shape[0]))
    print("Testing set has {} samples.".format(X_test.shape[0]))
    return X_train,X_test,y_train,y_test


#%%
"""
    Training Model - with or whithout tunning
    
"""
def model_train( model, parametros, X_train, y_train, tune=False):       
    
    if tune==False: #Trrain without parameters (without tunning)
        best_clf = model.fit(X_train, y_train)        
        
    else: #train tunning model
        scorer = make_scorer(f1_score, average='macro')
        # search best parameters and train model with them
        grid_obj = GridSearchCV(model, parametros, scoring=scorer)
        grid_fit = grid_obj.fit(X_train, y_train)        
        best_clf = grid_fit.best_estimator_    
        print('Best Param:', grid_obj.best_params_)
        
    return best_clf 
    


#%%
# Runs model pre-trained, using test set e returns metrics
def model_test(model, X_test, y_test):
    results = {}

    # predicting test
    pred_test = model.predict(X_test)
    
    # Metrics
    results['balanced_acc'] =  metrics.balanced_accuracy_score(y_test, pred_test)  
    results['f1'] = metrics.f1_score(y_test, pred_test, average=None)
    results['precision'] = metrics.precision_score(y_test, pred_test, average=None)
    results['recall'] = metrics.recall_score(y_test, pred_test, average=None)
    results['f1_macro'] = metrics.f1_score(y_test, pred_test, average='macro')
    results['precision_macro'] = metrics.precision_score(y_test, pred_test, average='macro')
    results['recall_macro'] = metrics.recall_score(y_test, pred_test, average='macro')
    
    print("Confusion Matrix:", metrics.confusion_matrix(y_test, pred_test))
     # Success
    print("{} trained .".format(model.__class__.__name__))
        
    # Return the results
    return results
    


#%%
    
# Trains Model, 
# techinique = 0 :None, 1:undersampling, 2: oversampling
# Runs and return results
def model_run(model, base, param, techinique=0, tune=False):
    
    X_train,X_test,y_train,y_test = train_test_sets(base, classes) #splits train and test
    
    if techinique==1: # NearMiss (undersampling)        
        nm = NearMiss()
        X_train, y_train = nm.fit_sample(X_train, y_train)
        print(np.bincount(y_train))
        
    elif techinique==2: # SMOTE (oversampling - )
        
        smt = SMOTE()
        X_train, y_train = smt.fit_sample(X_train, y_train)
        print( np.bincount(y_train))         
    
    # Train model
    model =  model_train( model, param, X_train, y_train, tune) 
    # Run model given
    results = model_test(model, X_test, y_test)
    return results

#%%
"""
    Pre processing and vectorizing data
"""
# Input Pre-Processing
tweets, classes = pp.input_treat()

# Vetorize tweets - using bag of words
#tweets_freq = pp.vetoriza(tweets, ngram_min=1,ngram_max=3)
tweets_freq = pp.vetoriza(tweets)


 #%%   
# Run model given
model_nb =  BernoulliNB()
model_svm = svm.SVC( decision_function_shape='ovo')

#%%

# Bernoulli - Without Tunning and don't resampling
results_nb = model_run(model_nb, tweets_freq,{ } ,0, False)


#%%
#  Create the parameters list you wish to tune, using a dictionary if needed.
parameters = {'alpha':[0.5, 1, 2],
              'binarize': [None, 0.2, 0.5, 0.7, 1.0],
              'fit_prior': [True, False]
              }



# Bernoulli - With Tunning and don't resampling
results_nb_tun = model_run(model_nb, tweets_freq, parameters , 0, True)

#%%
# SVM -Without Tunning and don't resampling
results_svm = model_run(model_svm, tweets_freq,{ } ,0, False)

#%%
# Parameters to SVM tune
params = {'C': [0.5, 1, 2] }

# Bernoulli - With Tunning and don't resampling
results_svm_tun = model_run(model_svm, tweets_freq, params ,0, True)

#%%
# com parâmetros otimizados
model_nb =  BernoulliNB()

# Bernoulli - Without Tunning and under-sampling
results_nb = model_run(model_nb, tweets_freq, parameters ,1, True)


#%%
#%%
# Parameters to SVM tune
params = {'C': [0.5, 1, 2] }

# Bernoulli - With Tunning and with resampling
results_svm_tun = model_run(model_svm, tweets_freq, params , 2, True)


#%%
'''import pandas as pd
df= pd.DataFrame(data=results_nb)
df['modelo'] = 'NB (default)'


df_2 = pd.DataFrame(data=results_nb_tun)
df_2['modelo'] = 'NB (best params)'


df_3 = pd.DataFrame(data=results_svm)
df_3['modelo'] = 'SVM (default)'

df_4 = pd.DataFrame(data=results_svm_tun)
df_4['modelo'] = 'SVM (best C)'


df_5 = pd.concat([df, df_2, df_3, df_4], ignore_index=True)
#%%%
from matplotlib import pyplot as plt

grupos = list( df_5['modelo'].unique() )
valores = list( df_5['balanced_acc'].unique( ))
#%%
plt.title("Acurácia Balanceada")
plt.bar(grupos, valores)
plt.show()
'''
#%%
