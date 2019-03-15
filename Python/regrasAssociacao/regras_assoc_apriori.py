# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 22:42:07 2019

@author: Sancha Azevedo
"""

import pandas as pd
from apyori import apriori


#  Função que Lê um arquivo de dados no formato CSV 
def le_arquivo():
    nome_arq = input("[caminho/]Nome do arquivo de dados[.extensão]:")
    arquivo = pd.read_csv(nome_arq, header=None) 
    return arquivo

#carrega o aquivo de dados
dados = le_arquivo()

# lista para receber as transações
transacoes =[]
# 6 transaççoes - Adiciona cada transação à lista 
for i in range (0,6):
    transacoes.append([str(dados.values[i,j]) for j in range(0,3)])
    
# regras de associação
regras = apriori(transacoes, min_support=0.5, min_confidence=0.5)    
resultado = list(regras)

# Visualizar as regras
resultados2 = [list(x) for x in resultado]


#dados = pd.read_csv("C:/Users/Sancha/Documents/IAeML/cientistaDadosRPython/machineLearning/dados/transacoes.txt", header=None )
