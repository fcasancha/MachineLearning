# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 16:38:33 2019

@author: Sancha
"""
from sklearn import datasets
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 

#carrega conjunto de dados default
iris = datasets.load_iris()
# Verifica quantos valores há por grupo, qts grupos
unicos, quantidade = np.unique(iris.target, return_counts =True)

# Agrupando com KMeans inicializadno com 3 grupos
cluster = KMeans(n_clusters=3)
cluster.fit(iris.data)

#visualizando resultado
centroides = cluster.cluster_centers_

#previsao: Em qual grupo cada instância vai ficar
previsoes = cluster.labels_
unicos2, quantidade2 = np.unique(previsoes, return_counts =True)

# Gerando a matriz de confusão para avaliar agrupamento - nesse caso não está na diagonal principal
resultados = confusion_matrix(iris.target, previsoes)

#plotar g´rafico de elemeentos do grpo zero. scatter:x,y, cor e rótulo
plt.scatter(iris.data[previsoes==0,0], iris.data[previsoes==0,1], c='green', label='Setosa')
plt.scatter(iris.data[previsoes==1,0], iris.data[previsoes==1,1], c='red', label='Versicolor')
plt.scatter(iris.data[previsoes==2,0], iris.data[previsoes==2,1], c='blue', label='Virgica')
plt.legend()
