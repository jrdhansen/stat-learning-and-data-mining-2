noise1 <- rnorm(n = 100, mean = 0, sd = 5)
noise2 <- rnorm(n = 100, mean = 0, sd = 5)
noise3 <- rnorm(n = 100, mean = 0, sd = 5)

x1 <- 1:100 + noise1
x2 <- 1:100 + noise2
x2 <- -x2
x3 <- 1:100 + noise3

plot(x1, x2)
plot(x3, x2)
plot(x1,x3)