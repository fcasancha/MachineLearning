# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 18:22:07 2019

@author: Sancha Azevedo
"""
import entrada_dados as ed
#pacote de machine learning-funcao para dividir a base entre treino e teste
from sklearn.model_selection import train_test_split 
#biblioteca de métricaas
from sklearn.metrics import confusion_matrix, accuracy_score
# Biblioteca com Árvore de Decisão
from sklearn.tree import DecisionTreeClassifier


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
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores,
                                                                  classe,
                                                                  test_size=0.3,
                                                                  random_state=0
                                                                  )


"""
    Criação do Modelo usando Árvore de Decisão
    passando para fit, o conj. de treino, e a saída do treino (classificação previa)
"""
arvore = DecisionTreeClassifier()
arvore.fit(X_treinamento, y_treinamento)

"""
    Previsão com Conj de Teste para Posterior Avaliação do Modelo
"""
previsoes = arvore.predict(X_teste)
previsoes

"""
    Avaliação do Modelo Criado
    Matriz de confusão dado o resultado de classificacao do teste e a 
    classificacao realizada
"""
confusao = confusion_matrix(y_teste, previsoes)
taxa_acerto = accuracy_score(y_teste, previsoes)
taxa_erro = 1 - taxa_acerto
