# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 20:09:13 2019
@author: Sancha
"""
import entrada_dados as ed
from sklearn.model_selection import train_test_split  #biblioteca para dividir treino e teste
from sklearn.naive_bayes import GaussianNB #gussian não trabalha com atributos categóricos, 
from sklearn.metrics import classification_report #gera relatórios com métrica F...
from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn.model_selection import cross_validate

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
atributos, atributo_classe = le_dados()

#   Construção do Modelo com Naive Bayes
#   Usando Hold Out ou Validação Cruzada
naiveBayes = GaussianNB() #modelo Nayve Bayes
quest = input(" Hold Out [HO] ou Validação Cruzada [VC]")
quest.upper()
if quest == "HO":

    # Técnica Hold Out - com 70% treino, 30% teste
    # Divisão do conjunto de treino e teste, dados:
    # os atributos, classe, %de teste, estado randomico 
    # 0 para conservar os mesmos conj caso rode várias vezes

    X_treino, X_teste, y_treino, y_teste = train_test_split(atributos,atributo_classe,
                                                           test_size=0.3, random_state=0)
    naiveBayes.fit(X_treino, y_treino)
    #   Previsão Teste pra avaliar o modelo treinado
    previsoes = naiveBayes.predict(X_teste)
    confusao = confusion_matrix(y_teste, previsoes)
    taxa_acerto = accuracy_score(y_teste, previsoes)
    taxa_erro = 1 - taxa_acerto
    print("Taxa Acerto:", taxa_acerto)
    print(classification_report(y_teste,previsoes))


    # Predição dos novos dados com modelo construído
    quest = input(" Deseja Classificar Dados Usando Modelo Construído [S/N]:")
    quest.upper()
    if quest=="S":
        atributos_2, atributo_classe_2 = le_dados()
        predicao2= naiveBayes.predict(atributos_2)
        print("Predição concluída... ")
        
elif quest=="VC":
    naiveBayes= cross_validate(naiveBayes, atributos, atributo_classe, cv=5)
     
else:
    print("Opção Inválida")  

    
    
#C:/Users/Sancha/Documents/IAeML/myCodes/BaseDados/classificacao_1.csv")
#C:/Users/Sancha/Documents/IAeML/cientistaDadosRPython/machineLearning/dados/Credit.csv