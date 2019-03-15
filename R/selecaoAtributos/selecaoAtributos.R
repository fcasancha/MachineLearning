# Pacote para selecao de atributos
install.packages("FSelector", dependencies = T)
library(FSelector) #carrega biblioteca para usar o Random Forest
library(e1071) #carrega biblioteca de classificadores, para usar o SVM

# carrega o arquivo de dados, com separador vírgula e cabeçalho
credito = read.csv(file.choose(), sep = ',', header = T)

# Selecao de atributo com o Random Forest
random.forest.importance(class ~., credito)

# Reutilizar o modelo criado no SVM, com os atributos de maior relevância
modelo_svm_2 = svm(class ~ checking_status + duration + credit_history + purpose + credit_amount, dados_treino)
#modelo_svm_2

# Fazer predicao com dados de teste para posterior avaliacao do modelo
predicao_2 = predict(modelo_svm_2, dados_teste)

# Matriz de confusao
# gera matriz de confusão - dados da classe previa, predicao
confusao = table(dados_teste$class, predicao_2)
confusao

#Taxa de Acerto
acerto = (confusao[1] + confusao[4]) / sum(confusao)
acerto

#taxa de Erro
erro = (confusao[2] + confusao[3]) / sum(confusao)
erro
