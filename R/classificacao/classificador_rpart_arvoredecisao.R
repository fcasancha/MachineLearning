# Árvore de Decisão com R Part - 
# Pode ser usado para Classificação ou para Regressão
install.packages("rpart", dependencies = T) # Instalação da biblioteca rpart
library(rpart) # Carrega biblioteca

# carrega arquivo de dados que serão classificados, separação por vírgula e cabeçalho True
credito = read.csv(file.choose(), sep = ',', header = T)

# Separar Dados de treino e teste para posterior construção e avaliação do modelo
# Parâmetros: 2 conjuntos, 1000 entradas, com reposição,  % de treino e teste respectivamente
amostra = sample( 2, 1000, replace = T, prob = c(0.7,0.3) )
conj_treino = credito[amostra==1,]
conj_teste = credito[amostra==2,]

# Criação da Árvore de Decisão - Modelo de classificação
# Parâmetros: atributo de classificação e os demais (. = todos), 
#    os dados de treino, método pode ser: classificação ou regressão
arvore = rpart(class ~., data=conj_treino, method="class")

print(arvore) #impressão da árvore gerada
# Impressão da árvore gerada de forma gráfica
plot(arvore)
text(arvore, use.n = TRUE, all = TRUE, cex=.8) #aparecer valores nas folhas

# Predição do conjunto de teste - Probabilidade por classe
# Parâmetros: modelo (árvore), conjunto a ser classificado 
teste_classif = predict(arvore, newdata = conj_teste )
teste_classif

# Converter probalidade para valor binário (bom ou exclusivo mau)
# Cbind add uma coluna em cred
# Parâmetros: conj de teste, dados de predição realizada sobre o conj. teste
cred = cbind(conj_teste, teste_classif)
fix(cred) #mostra o resultado de cred
cred['Result'] = ifelse(cred$bad >=0.5, "bad", "good")

# matriz de confusão
confusao = table(cred$class, cred$Result)
confusao

# Taxas de Acerto e de Erro
taxaacerto = (confusao[1] + confusao[4]) / sum(confusao)
taxaacerto
taxaerro = (confusao[2] + confusao[3]) / sum(confusao)
taxaerro


