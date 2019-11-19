# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:12:57 2019

@author: Sancha Azevedo

Geração de amostra de dados simples com reposição
"""

import pandas as pd
import numpy as np

#Carrega o arquivo de dados para o dataframe base
base = pd.read_csv('iris.csv')
base.shape

# Inserindo a semente para gerar os mesmos valores se a função randomica for executada várias vezes
np.random.seed(2345)

# amostra com prob. de gerar 50% de 0's e 50% 1's
amostra = np.random.choice(a = [0,1], size = 150, replace=True, p=[0.5,0.5])
len(amostra[amostra==1])
len(amostra[amostra==0])

