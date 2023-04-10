options(scipen = 999)

library("mvtnorm")
library("rstan")
library("mlogit")
source("data_generation.R")
library("stringr")
library("glue")

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# --------------------------
# Генерация данных
# --------------------------

# Пример описания процесса генерации данных и ошибок в файле "theory_data.R"
name_exp <- "init"      # Лейбл эксперимента
li_type <- "linear"     # Тип линейного индекса
error_type <- "normal"  # Тип генерации ошибки
sigma_1 <- 1            # Параметры ковариационной матрицы
sigma_2 <- 3
sigma_3 <- 4
corr_12 <- 0.4
corr_13 <- -0.6
corr_23 <- 0.3

n_array <- c(25, 50, 100, 500, 1000, 5000)  # Число наблюдений
iter_all <- 50                              # Число итераций

for (n in n_array) {
  for (num_iter in 1:iter_all){
    
    # 1. Генерация данных
    h <- data_generate(n, name_exp, num_iter, li_type, error_type, 
                       sigma_1, sigma_2, sigma_3, corr_12, corr_13, corr_23)
    
    # 2. Оценивание модели ММП
    h_ml <- dfidx(h,                                  
                  shape = "wide",                 
                  choice = "transport")   
    
    model_cmprobit <- mlogit(transport ~ 1 | income + health + age, 
                             data = h_ml, 
                             probit = TRUE)               
    
    coef_cmprobit <- coef(model_cmprobit,)           
    probs_cmprobit <- predict(model_cmprobit,
                              newdata = h_ml)
    
    ### Сохранение результатов
    directory <- glue("results\\exp_{name_exp}\\n_{n}\\")
    dir.create(directory)
    
    write.csv(coef_cmprobit, 
              glue("{directory}\\iter_{num_iter}_coefficients_ml.csv"), 
              row.names=FALSE)
    
    write.csv(probs_cmprobit, 
              glue("{directory}\\iter_{num_iter}_probabilities_ml.csv"), 
              row.names=FALSE)
    
    # 3. Оценивание модели байесовским методом
    
  }
}

h

data <- list(x_1 = X_1,
             x_2 = X_2,
             y = y,
             n = dim(X_1)[1],                 # int<lower=0> n
             k_1 = dim(X_1)[2],               # int<lower=0> k
             k_2 = dim(X_2)[2])

model_normal <- stan(file = "Bivariate probit model with normal errors.stan",
                     data = data,                # входные данные
                     chains = 1,                 # количество выборок из апостериорного распределения
                     iter = 2000)                # удвоенный объем выборки из апостериорного распределения
posterior_normal <- extract(model_normal)

df_normal <- estimations_metrics(df=df_normal, posterior=posterior_normal, k_1=dim(X_1)[2], k_2=dim(X_2)[2])

names(df_normal) <- rnames

true = c(beta_1, beta_2, sigma[1,2] / (sqrt(sigma[1,1] * sigma[2,2])), rep(0, length(rnames)-length(beta_1)-length(beta_2)-1))

table = data.frame(true=true,
#                   normal=colMeans(df_normal))
