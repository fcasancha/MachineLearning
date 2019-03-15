# Seleção de Atributos usando o SVM (Suporte Vector Machine)
# Usa o pacote e1071

#install.packages("e1071", dependencies=T)
library(e1071)

credito = read.csv(file.choose(), sep=",", header = T)
# 70% treino e 30% teste, universo 2, 1000 sorteios de 1 ou 2, replace=reposição, probabilidade
amostra = sample( 2, 1000, replace = T, prob = c( 0.7,0.3) ) 
amostra #imprime

# Probabilidade em amostra 70% de ser 1 e 30% de ser2
dados_treino = credito[amostra==1,]
dados_teste = credito[amostra==2,]


# Criar modelo Usando o SVM
modelo_svm = svm(class ~., dados_treino)
modelo_svm

# Predição com modelo gerado e ddos de teste
predicao = predict(modelo_svm, dados_teste)

# gera matriz de confusão - dados da classe previa, predicao
confusao = table(dados_teste$class, predicao)
confusao

#Taxa de Acerto
acerto = (confusao[1] + confusao[4]) / sum(confusao)
acerto

#taxa de Erro
erro = (confusao[2] + confusao[3]) / sum(confusao)
erro

