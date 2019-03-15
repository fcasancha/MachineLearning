# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 18:03:51 2019

@author: Sancha
"""

import entrada_dados as ed
#pacote de machine learning-funcao para dividir a base entre treino e teste
from sklearn.model_selection import train_test_split 
# Métricas
from sklearn.metrics import confusion_matrix, accuracy_score
# Biblioteca com o Classificador Random Forest
from sklearn.ensemble import RandomForestClassifier


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


"""
    Divisão da base em treino e teste
"""
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores,classe,
                                                                  test_size=0.3,
                                                                  random_state=0
                                                                  )
"""
    Cria floresta com 100 árvores para fazer a votação e com base nisto, classificar
"""
floresta = RandomForestClassifier(n_estimators=100)
floresta.fit(X_treinamento, y_treinamento)
previsoes = floresta.predict(X_teste)
confusao = confusion_matrix(y_teste, previsoes)
taxa_acerto = accuracy_score(y_teste, previsoes)

floresta.estimators_
#visualiza 1 arvore
floresta.estimators_[0]