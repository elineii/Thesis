options(scipen = 999)

library("mvtnorm")
library("rstan")
library("mlogit")
source("data_generation.R")
library("stringr")
library("glue")

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
set.seed(42) 

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
iter_all <- 1                               # Число итераций

for (n in n_array[1:1]) {
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
    
    write.csv(full_h, 
              glue("{directory}\\coefficients_ml.csv"), 
              row.names=FALSE,
              append=TRUE)
    
    write.csv(full_h, 
              glue("{directory}\\iter_{num_iter}_probabilities_ml.csv"), 
              row.names=FALSE)
    
    # 3. Оценивание модели байесовским методом
    
  }
}
