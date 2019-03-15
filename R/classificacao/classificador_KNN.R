# Instalação do pacote com conjunto de dados Iris
install.packages("class", dependencies = T)
library(class)
head(iris)

library(e1071) #biblioteca de classificadores

# Divisao do conjunto de dados em 2, de 150 instâncias, com reposição, probabilidade 70 e 30%
amostra = sample(2, 150, replace = T, prob = c(0.7, 0.3))
iris_treino = iris[amostra==1, ]
iris_teste = iris[amostra==2, ]

# Previsão com base nas instâncias
# Parâmetros: 
# 1o Vizinhança: conj de treino, com colunas de 1 a 4, excluindo a 5a previamente classificada
# 2o a classificar: iris_teste, 
# 3o classes da vizinhança
# 4o k
#previsao = knn(iris_treino[, 1:4], iris_teste[, 1:4], iris_treino[, 5], k=3)
previsao = knn(iris_treino[,1:4], iris_teste[,1:4], iris_treino[,5], k=3)
table(iris_teste[,5], previsao)

