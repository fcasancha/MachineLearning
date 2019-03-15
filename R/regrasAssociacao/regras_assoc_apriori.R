install.packages("arules")

library(arules)
#cada compra em uma linha, com atrib sep por virgula
transacoes = read.transactions(file.choose(), format = "basket", sep=",")
transacoes

# estrutura do arquivo carregado
inspect(transacoes)
# gráfico do arquivo de transações
image(transacoes)

# Criando Regras de associaçao -param:dados_transacoes, suporte e confiança mínimos pra gerar as regras 
regras = apriori(transacoes, parameter = list(supp=0.5, conf=0.5))

# Ver regras geradas pelo apriori
inspect(regras)