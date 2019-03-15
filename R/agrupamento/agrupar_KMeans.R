#Utiliza Kmens passando os dados e qtd de grupos
cluster = kmeans(iris[1:4],center=3)
cluster

table(iris$Species,cluster$cluster)
plot(iris[,1:4],col=cluster$cluster)
