# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 18:53:40 2019

@author: Sancha Azevedo
"""
# Biblioteca para ler arquivo e transformar dados categóricos em numéricos
import entrada_dados as ed
#pacote de machine learning-funcao para dividir a base entre treino e teste
from sklearn.model_selection import train_test_split 
# Biblioteca de métricaas
from sklearn.metrics import confusion_matrix, accuracy_score
# SVM
from sklearn.svm import SVC
# Algoritmo de floresta randomica
from sklearn.ensemble import ExtraTreesClassifier 

#  Função para chamar a leitura de arquivo e tratar caso seja necessário (usando funções do módulo)
def le_dados():
    # chama função para fazer leitura do arquivo de dados
    arquivo = ed.le_arquivo()
    # Verifica se precisa separar dados da classe - aplica-se a construção do modelo
    quest = input(" Separar atributos de atributo Classe:[S/N]")
    quest.upper()
    if quest == 'S':
        atributos, atributo_classe = ed.separa_atributos(arquivo, True)    
    else:
        atributos, atributo_classe = ed.separa_atributos(arquivo, False)
    
    # Verifica se os dados precisam sem transformados em numéricos
    quest = input(" Precisa transformar dados categóricos em numéricos:[S/N]")
    quest.upper()
    if quest == 'S':
        atributos = ed.para_campos_numericos(atributos)
    
    return atributos, atributo_classe

#  Leitura dos dados de entrada
previsores, classe = le_dados()

# Dividindo arquivo lido em 70% treino e 30% teste
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores, classe, 
                                                        test_size=0.3, random_state=0
                                                  )

"""
    Construindo o modelo , classificando teste e avaliando
"""
modelo = SVC()
modelo.fit(X_treinamento, y_treinamento)
previsoes = modelo.predict(X_teste)
taxa_acerto = accuracy_score(y_teste, previsoes)

"""
    Avaliar a importância dos atributos
"""
forest = ExtraTreesClassifier()
forest.fit(X_treinamento, y_treinamento) #modelo usando floresta
importancia = forest.feature_importances_


"""
    Construindo modelo com os 4 primeiros atributos mais relevantes
""" 
X_treinamento_2 = X_treinamento[:,[0,1,2,4]]
X_teste_2 = X_teste[:,[0,1,2,4]]
modelo_2 = SVC()
modelo_2.fit(X_treinamento_2, y_treinamento)  #modelo 
previsoes_2 = modelo_2.predict(X_teste_2) #previsao do conjunto de teste
taxa_acerto_2 = accuracy_score(y_teste, previsoes_2) #acurácia da previsão
    