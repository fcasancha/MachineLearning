# Fazer previsao de novos dados (Produção), utilizando modelo previamente criado

# carregar dados para previsao
novocredito = read.csv(file.choose(), sep=',', header=T)

novocredito
dim(novocredito)

#predição, dados o modelo e os dados novos
predict(modelo, novocredito)