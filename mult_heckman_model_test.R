options(scipen = 999)

library("mvtnorm")
library("rstan")
library("mlogit")
source("data_generation.R")
library("stringr")
library("glue")

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

options(scipen = 999)
set.seed(123)

df <- read.csv("data/simulation_benchmark/n_100/sim_1_n_100.csv")
df$y[is.na(df$y)] <- (-1)


data <- list(n = 100,
             k_1 = 3,
             k_2 = 3,
             w = data.matrix(df[c("w0", "w1", "w2")]),
             x = data.matrix(df[c("x0", "x1", "x2")]),
             y = df$y)

model_normal <- stan(file = "stan_models/multinomial_heckman.stan",
                     data = data,                # входные данные
                     chains = 1,                 # количество выборок из апостериорного распределения
                     iter = 2000)                # удвоенный объем выборки из апостериорного распределения

posterior_normal <- extract(model_normal)

