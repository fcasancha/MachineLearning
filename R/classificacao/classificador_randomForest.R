# Instalação do pacote Random Forest
install.packages('randomForest',dependencies=T)
library(randomForest)

credito = read.csv(file.choose(), sep=",", header = T)
# 70% treino e 30% teste, universo 2, 1000 sorteios de 1 ou 2, replace=reposição, probabilidade
amostra = sample( 2, 1000, replace = T, prob = c( 0.7,0.3) ) 
amostra #imprime

# Probabilidade em amostra 70% de ser 1 e 30% de ser2
dados_treino = credito[amostra==1,]
dados_teste = credito[amostra==2,]

# Construção do Modelo com Random Forest
#floresta = randomForest(class ~.,data=dados_treino, ntree=100, importance=T)
floresta = randomForest(class ~ .,data=creditotreino, ntree=100,importance=T)
varImpPlot(floresta)

# Teste com modelo criado
previsao = predict(floresta, dados_teste)
confusao = table(previsao, dados_teste$class)
confusao

#taxas de acerto e erro
taxaacerto = (confusao[1] + confusao[4]) / sum(confusao)
taxaerro = (confusao[2] + confusao[3]) / sum(confusao)