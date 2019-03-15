# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:58:41 2019

@author: Sancha

    Módulo com funções para:
        -Leitura de arquivo em formato csv 
        -Separar atributos da classe /atributo alvo (no treino)
        -Transformar atributos categóricos em numéricos

"""
# Bibliotecas
from sklearn.preprocessing import LabelEncoder
import pandas as pd  #biblioteca de manipulação de dados 


#  Função que Lê um arquivo de dados no formato CSV 
def le_arquivo():
    nome_arq = input("[caminho/]Nome do arquivo de dados[.extensão]:")
    arquivo = pd.read_csv(nome_arq) 
    #arquivo.head()    
    return arquivo


# Função que Transforma cada Atributo Categórico (na pos dada) em Numérico
def para_campos_numericos(atributos):
    dados = atributos
    qtd = int(input("Quantos atributos deseja tratar:"))
    labelencoder = LabelEncoder()
    
    for i in range(qtd):
        print("coluna:", i)
        dados[:,i]= labelencoder.fit_transform(dados[:,i])
    return dados     

# Função que recupera os dados do arquivo pelas linhas e índices das colunas 
# Se treino_teste verdadeiro, devolve atributos e atributo de classificação separadamente
# senão retorna todas as linhas e colunas recuperadas do arquivo 
def separa_atributos(arquivo, treino_teste):
    #  Atributos (exceto classe) 
    #  Atributo de Classificação - último atributo do dataframe
    qtde_atributos  = int(input("Informe a quantidade de atributos (total):"))
    if treino_teste == True:
        atributos = arquivo.iloc[:, 0:(qtde_atributos-1)].values
        atributo_classe = arquivo.iloc[:, (qtde_atributos-1)].values
    else:
        atributos = arquivo.iloc[:, 0:(qtde_atributos)].values
        atributo_classe = []
    
    return atributos, atributo_classe

