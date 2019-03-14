# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 23:12:35 2019

@author: Sancha
"""

import entrada_dados as ed
#pacote de machine learning-funcao para dividir a base entre treino e teste
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, accuracy_score
# Biblioteca do KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


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

"""Divisão da base em treino e teste
 Saída: X_train, X_test, y_train, y_test são arrays da NumPy
 X_train: 70% do conj de dados
 X_test: 30%
 Y: serão as repostas para os respectivos conjuntos
 Parametros: attr, classe, %teeste, 
 random_state para manter a mesma saída se executar a mesma função várias vzs
"""
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores, classe,
                                                                  test_size=0.3,
                                                                  random_state=0
                                                                  )
"""
    Classificação com KNN
    n_neighbors = Quantidade de classes a classificar   
"""
qtd_clas = input(" Quantidade de classes:")
qtd_classes = int(qtd_clas)
knn = KNeighborsClassifier(n_neighbors=qtd_classes)
knn.fit(X_treinamento, y_treinamento)
previsoes = knn.predict(X_teste)
confusao = confusion_matrix(y_teste, previsoes)
taxa_acerto = accuracy_score(y_teste, previsoes)
taxa_erro = 1 - taxa_acerto

print(classification_report(y_teste,previsoes))