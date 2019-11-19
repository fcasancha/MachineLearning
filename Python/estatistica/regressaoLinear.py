# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 19:15:45 2019

@author: Sancha Azevedo
"""
# Importação das bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Importação (leitura da base de dados)
dados = pd.read_csv("C:/Users/Sancha/Documents/IAeML/myCodes/BaseDados/cars.csv")
#remove a coluna Unnamed: 0, eixo: coluna, e altera no dataframe (inplace=True)
dados.drop("Unnamed: 0", axis=1, inplace=True)

# Verificar se há correlação entre as duas features: speed e dist
# X: variável dependente dist (formato numpy array)
# y: variável a ser predita (independente): speed
x = dados.iloc[:,1].values
y = dados.iloc[:, 0].values

# coef. de correlação em matriz (+próx de 1->forte)
correlacao = np.corrcoef(x, y)

#formatando os dados para entrada do modelo de regressão
x = x.reshape(-1,1) 
# se a correlação for forte, é aplicável a regressão linear
modelo = LinearRegression()
modelo.fit(x,y) #treino

inter = modelo.intercept_
coef = modelo.coef_

# Plotando gráfico com linha de regressão
plt.scatter(x, y)
plt.plot(x, modelo.predict(x), color="red")

# predizendo a velocidade com a distância 22, usando a fórmula
inter + (coef * 22)

#predizendo com a função de predição do modelo
valor = np.array(22)
valor = valor.reshape(-1,1)
modelo.predict(valor)


# Residuais: diferença dos pontos do modelo para a linha de regressão
#modelo.residues