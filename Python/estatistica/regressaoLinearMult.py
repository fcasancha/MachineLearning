# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:46:41 2019

@author: Sancha Azevedo
"""
# Importação das bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

# Importação (leitura da base de dados)
dados = pd.read_csv("C:/Users/Sancha/Documents/IAeML/myCodes/BaseDados/mt_cars.csv")
#remove a coluna Unnamed: 0, eixo: coluna, e altera no dataframe (inplace=True)
dados.drop("Unnamed: 0", axis=1, inplace=True)

# Verificar se há correlação entre as duas features: speed e dist
# X: variável dependente dist (formato numpy array)
# y: variável a ser predita (independente): speed
x = dados.iloc[:,2].values
y = dados.iloc[:, 0].values

# coef. de correlação em matriz (+próx de 1->forte)
correlacao = np.corrcoef(x, y)

#formatando os dados para entrada do modelo de regressão
x = x.reshape(-1,1) 
# se a correlação for forte, é aplicável a regressão linear
modelo = LinearRegression()
modelo.fit(x,y) #treino

# interceptação, inclinação 
inter = modelo.intercept_
coef = modelo.coef_
#R2
score = modelo.score(x,y)

# previsão dda base de treino
previsoes = modelo.predict(x)
modelo_ajustado = sm.ols(formula='mpg ~ disp', data=dados)
modelo_treinado = modelo_ajustado.fit()
modelo_treinado.summary()

# plotagem de gráfico 
plt.scatter(x,y)
plt.plot(x, previsoes, color='red')


# preedizendo dado novo
valor = np.array(200)
valor = valor.reshape(-1,1)
modelo.predict(valor)

# Regressão Linear Múltipla
x1 = dados.iloc[:,1:4].values
y1 = dados.iloc[:, 0].values
modelo2 = LinearRegression()
modelo2.fit(x1, y1)

modelo2.score(x1, y1)
modelo_ajustado2 = sm.ols('mpg ~ cyl + disp + hp', data = dados)
modelo_treinado2 = modelo_ajustado2.fit()
modelo_treinado2.summary()

# predizendo dado novo
novo = np.array([4, 200, 100])
novo = novo.reshape(1, -1)
modelo2.predict(novo)