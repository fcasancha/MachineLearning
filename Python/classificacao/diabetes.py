# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:45:55 2020

@author: Sancha Azevedo
"""
#%%
# Bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import seaborn as sns
sns.set(style="ticks")


#%%
###########################################
#       Análise Exploratória dos dados
###########################################

# Lendo o dataset diabetes
diabetes = pd.read_csv('diabetes.csv')
#diabetes = pd.read_csv('C:/Users/Sancha/Documents/IAeML/myCodes/BaseDados/diabetes.csv')
# Visualizando as primeiras linhas dataset
diabetes.head()

#%%
# Obtendo mais informaçoes e a correlação das features com outcome
diabetes.info()
diabetes.corr()

#%%
# Verificando percentual de diabéticos da base de dados
total_diab  = diabetes.Outcome.sum()
print('% de Diabéticos na base:')
print(total_diab/len(diabetes))

#%%
# Plotando Histograma dos atributos
def histograma(atributo, dataframe):
    plt.style.use('seaborn-white')
    plt.title(atributo)
    plt.hist(list(dataframe[atributo]));
    

histograma('Age', diabetes)

#%%
histograma('Glucose', diabetes)

#%%
histograma('BloodPressure', diabetes)

#%%
histograma('Insulin', diabetes)

#%%
histograma('BMI', diabetes)
    
#%%
###########################################
#      Separando dados para treino e teste
###########################################
y = diabetes['Outcome']
X = diabetes.iloc[:, 0:9]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
###########################################
#      Construindo classificadores
###########################################

def classificador(clf, param, X_train, y_train, X_test, nome_class):
# Executa uma busca randomizada sobre os hiper parâmetros
    random_search = RandomizedSearchCV(clf, param_distributions=param)

    # Treina o modelo usando o conjunto de treino
    random_search.fit(X_train, y_train)

    # parãmetros melhor estimador
    best = random_search.best_estimator_

    # Executa predição sobre conjunto de teste
    preds = best.predict(X_test)
    # Métricas de avaliação
    confusao = confusion_matrix(y_test, preds)
    acuracia = accuracy_score(y_test, preds)
    precisao = precision_score(y_test, preds)
    revocacao = recall_score(y_test, preds)
    f = f1_score(y_test, preds)
    print('-----------')
    print(nome_class)
    print('-----------')
    print(' Melhores preditores', best)
    print('Confusão:', confusao, ' \nAcurácia:', acuracia, ' Precisão:',precisao, ' Revocacao:',revocacao, ' F:',f)

#%%
# Random Forest
clf_rf = RandomForestClassifier()

#  hyperparametros 
param_dist = {"max_depth": [3, 4, 5, 6],
              "n_estimators": [10, 50, 100, 500, 1000],
              "max_features": list(range(1, X_test.shape[1]+1)),
              "min_samples_split": [50, 80, 100],
              "min_samples_leaf": [5, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

classificador(clf_rf, param_dist, X_train, y_train, X_test, 'Random Forest')

#%%
# Ada Boost
clf_ab = AdaBoostClassifier()
param_ab = {'n_estimators':[100, 500, 1000, 10000], 'learning_rate':[0.1, 0.5, 0.8, 1]}
classificador(clf_ab, param_ab, X_train, y_train, X_test, 'Ada Boost')

#%%
# SVM
clf_svm = SVC()
param_svm = {'C':[0.1, 0.5, 0.8, 1, 10], 'kernel':['linear','rbf','poly','sigmoid']}

classificador(clf_svm, param_svm, X_train, y_train, X_test, 'SVM')



