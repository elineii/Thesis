options(scipen = 999)
set.seed(123)

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

df <- read.csv("data/simulation_benchmark/n_50/sim_1_n_50.csv")
df$y[is.na(df$y)] <- Inf

# MLE model
model <- mnprobit(z ~ w1 + w2,
                  y ~ x1 + x2,
                  regimes = c(0, -1, -1),
                  data = df, cov_type = "gop")
summary(model)
# TODO: ВЫТАЩИТЬ МАТ ОЖИДАНИЕ И СТАНДАРТНОЕ ОТКЛОНЕНИЕ ГАММ И БЕТЫ
res <- model$tbl$coef2[[1]][1:3]
res <- data.frame(beta_0=res[1], beta_1=res[2], beta_3=res[3])
beta_exp_ml <- 
beta_exp_sd <-
gamma_1_exp_ml <- 
gamma_1_sd_ml <- 
gamma_2_exp_ml <- 
gamma_2_sd_ml <-

# Stan model with fixed params
model <- stan(file = "stan_models/multinomial_heckman_debug_likelihood.stan",
              algorithm = "Fixed_param",
              data = data,                # входные данные
              chains = 1,                 # количество выборок из апостериорного распределения
              iter = 2000)                # удвоенный объем выборки из апостериорного распределения
posterior <- extract(model)

# Stan model with params from MLE
data <- list(n = dim(df)[1],
                k_1 = 3,
                k_2 = 3,
                w = data.matrix(df[c("w0", "w1", "w2")]),
                x = data.matrix(df[c("x0", "x1", "x2")]),
                z = df$z,
                y = df$y,
                beta_exp_ml = beta_exp_ml,
                beta_exp_ml = beta_exp_ml,
                beta_exp_sd = beta_exp_sd,
                gamma_1_exp_ml = gamma_1_exp_ml,
                gamma_1_sd_ml  = gamma_1_sd_ml, 
                gamma_2_exp_ml = gamma_2_exp_ml,
                gamma_2_sd_ml  = gamma_2_sd_ml)

model <- stan(file = "stan_models/multinomial_heckman.stan",
                     data = data,                # входные данные
                     chains = 1,                 # количество выборок из апостериорного распределения
                     iter = 2000)                # удвоенный объем выборки из апостериорного распределения
posterior <- extract(model)
