#install.packages("arules")
library(arules)

#cada compra em uma linha, com atrib sep por virgula
transacoes = read.transactions(file.choose(), format = "basket", sep=",")
transacoes

# Mineração das regras de associações
# enviando dados de transações, em parametrer: suporte e max de itens
regras = eclat(transacoes, parameter = list(supp=0.1, maxlen=15))

# Visualizando as regras criadas
inspect(regras)

