# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:19:31 2019

@author: Sancha Azevedo

Amostragem estratificada
"""
import pandas as pd
from sklearn.model_selection import train_test_split

iris = pd.read_csv('iris.csv')
#conta as instâncias de acordo com a classe
iris['class'].value_counts()

# dados de entrada, classe, %de split, estratificar pela classe (atrib 4)
x, _, y, _ = train_test_split(iris.iloc[:,0:4], iris.iloc[:,4], test_size=0.5,
                        stratify=iris.iloc[:, 4])
y.value_counts()


# outra base de dados..
# 12/248*100 -> 1 classe
# e assim sucessivamente...
infert = pd.read_csv('C:/Users/Sancha/Documents/IAeML/myCodes/Python/estatistica/infert.csv')
infert['education'].value_counts()
# dados de entrada, classe, %de split, estratificar pela classe (atrib 4)
x1, _, y1, _ = train_test_split(infert.iloc[:,2:9], infert.iloc[:,1], test_size=0.6,
                        stratify=infert.iloc[:, 1])

y1.value_counts()   