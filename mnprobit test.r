# Packages
library("matrixcalc")
library("mnorm")
library("switchSelection")

# Set seed for reproducibility
set.seed(123)

# The number of observations
n <- 10000

# Covariance matrix
sigma2 <- sqrt(1.2)
rho12 <- 0.5
sigma.y0 <- sqrt(2)
rho1.y0 <- -0.3
rho2.y0 <- 0.5
sigma.y1 <- sqrt(1.8)
rho1.y1 <- -0.4
rho2.y1 <- 0.2
rho.y <- 0.6
sigma <- matrix(c(1,                  rho12 * sigma2,              rho1.y0 * sigma.y0,          rho1.y1 * sigma.y1,
                  rho12 * sigma2,     sigma2 ^ 2,                  rho2.y0 * sigma2 * sigma.y0, rho2.y1 * sigma2 * sigma.y1,
                  rho1.y0 * sigma.y0, rho2.y0 * sigma2 * sigma.y0, sigma.y0 ^ 2,                rho.y * sigma.y0 * sigma.y1,
                  rho1.y1 * sigma.y1, rho2.y1 * sigma2 * sigma.y1, rho.y * sigma.y0 * sigma.y1, sigma.y1 ^ 2),
                ncol = 4, byrow = TRUE)
colnames(sigma) <- c("z1", "z2", "y0", "y1")
rownames(sigma) <- colnames(sigma)
is.positive.definite(sigma)

# Simulate random errors
errors <- rmnorm(n, c(0, 0, 0, 0), sigma)
u <- errors[, 1:2]
eps <- errors[, 3:4]

# Regressors
x1 <- runif(n, -1, 1)
x2 <- runif(n, -1, 1)
X <- cbind(1, x1, x2)

# Coefficients
gamma <- cbind(c(0.1, 1, 1), c(0.2, -1, 0.5))
beta <- matrix(c(1, -1, 1, 
                 1, 1, -1),
               nrow = 2, byrow = TRUE)

# Linear indexes
z1.li <- X %*% gamma[, 1]
z2.li <- X %*% gamma[, 2]
y0.li <- X %*% t(beta[1, , drop = FALSE])
y1.li <- X %*% t(beta[2, , drop = FALSE])

# Latent variables
z1.star <- z1.li + u[, 1]
z2.star <- z2.li + u[, 2]
y0.star <- y0.li + eps[, 1]
y1.star <- y1.li + eps[, 2]

# Obvservable variable as a dummy
z1 <- (z1.star > z2.star) & (z1.star > 0)
z2 <- (z2.star > z1.star) & (z2.star > 0)
z3 <- (z1 != 1) & (z2 != 1)

# Aggregate observable variable
z <- rep(0, n)
z[z1] <- 1
z[z2] <- 2
z[z3] <- 3
table(z)

# Make unobservable values of continuous variable
y <-  rep(Inf, n)
y[z == 2] <- y0.star[z == 2]
y[z == 3] <- y1.star[z == 3]

# Data
data <- data.frame(z = z, y = y, 
                   x1 = x1, x2 = x2)

# MLE
model <- mnprobit(z ~ x1 + x2, 
                  y ~ x1 + x2, regimes = c(-1, 0, 1),
                  data = data, cov_type = "no")
summary(model)
model$coef

# Estimation
model <- mnprobit(z ~ x1 + x2, data = data)
summary(model)
lambda1 <- predict(model, type = "lambda", alt = 1)
lambda2 <- predict(model, type = "lambda", alt = 2)
lambda3 <- predict(model, type = "lambda", alt = 3)
data$lambda1[z == 1] <- lambda1[z == 1, 1] + lambda1[z == 1, 2]
data$lambda2[z == 1] <- -lambda1[z == 1, 1]
data$lambda1[z == 2] <- -lambda2[z == 2, 1]
data$lambda2[z == 2] <- lambda2[z == 2, 1] + lambda2[z == 2, 2]
data$lambda1[z == 3] <- -lambda3[z == 3, 1]
data$lambda2[z == 3] <- -lambda3[z == 3, 2]
model2 <- lm(y ~ x1 + x2 + lambda1 + lambda2,
             data = data[!is.infinite(y), ])
summary(model2)
sigma[1, 3]
sigma[2, 3]
