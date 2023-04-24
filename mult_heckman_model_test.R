options(scipen = 999)

library("mvtnorm")
library("rstan")
library("mlogit")
source("data_generation.R")
library("stringr")
library("glue")
library("mnorm")
library("switchSelection")

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

options(scipen = 999)
set.seed(123)

df <- read.csv("data/simulation_benchmark/n_50/sim_1_n_50.csv")
df$y[is.na(df$y)] <- Inf

data <- list(n = 50,
             k_1 = 3,
             k_2 = 3,
             w = data.matrix(df[c("w0", "w1", "w2")]),
             x = data.matrix(df[c("x0", "x1", "x2")]),
             z = df$z,
             y = df$y)

model <- stan(file = "stan_models/multinomial_heckman.stan",
                     data = data,                # входные данные
                     chains = 1,                 # количество выборок из апостериорного распределения
                     iter = 2000)                # удвоенный объем выборки из апостериорного распределения

posterior <- extract(model)

model <- stan_model(file = "stan_models/multinomial_heckman_debug_likelihood.stan")
f <- optimizing(model, data = data,
           init = 'random')


model <- mnprobit(z ~ w1 + w2,
                  y ~ x1 + x2,
                  regimes = c(0, -1, -1),
                  data = df, cov_type = "gop")
summary(model)

model <- stan(file = "stan_models/multinomial_heckman_debug_likelihood.stan",
              algorithm = "Fixed_param",
              data = data,                # входные данные
              chains = 1,                 # количество выборок из апостериорного распределения
              iter = 2000)                # удвоенный объем выборки из апостериорного распределения

posterior <- extract(model)

Log-likelihood = -18549.0227