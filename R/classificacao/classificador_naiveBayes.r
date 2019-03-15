# HÁ 2 PACOTES PARA O NAYVE BAYES: el071 e klaR
install.packages("e1071", dependencies=T)

# carrega o pacote Nayve Bayes e171
library(e1071)


# carregando dados do arquivo escolhido, separador é a vírgula e tem cabeçalho (nomes nos atributos)
credito = read.csv(file.choose(), sep=",", header = T)

head(credito) #imprime as primeiras instâncias
dim(credito) #dimensões do credito

# 70% treino e 30% teste, universo 2, 1000 sorteios de 1 ou 2, replace=reposição, probabilidade
amostra = sample( 2, 1000, replace = T, prob = c( 0.7,0.3) ) 
amostra #imprime

# Probabilidade em amostra 70% de ser 1 e 30% de ser2
dados_treino = credito[amostra==1,]
dados_teste = credito[amostra==2,]

# dimensões dos conjuntos de treino e teste
dim(dados_treino)
dim(dados_teste)

# Criação do Modelo uhuuu :) 
# pode ser savo em disco, persistir no R, integrar com outra aplicação, ...
# Parametros: 1o - Atributos, varResposta~VarsExplicativas (. são todos atributos)
# 2o dados de treino
modelo = naiveBayes(class ~., dados_treino)
modelo

# Testando o modelo com dados de teste para análise do modelo
# Função predict
# Parâmetros: modelo criado, dados para predição
predicao = predict(modelo, dados_teste)
predicao

#gera matriz de confusão para avaliar o modelo 
confusao = table(dados_teste$class, predicao)
confusao

#Taxa de Acerto
acerto = (confusao[1] + confusao[4]) / sum(confusao)
acerto

#taxa de Erro
erro = (confusao[2] + confusao[3]) / sum(confusao)
erro