options(scipen = 999)

library("mvtnorm")
library("rstan")
library("mlogit")
source("C:\\Users\\S\\Documents\\Education\\Lina\\Thesis\\data_generation.R")
library("stringr")
library("glue")
library("mltools")
library("data.table")

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
sigma_2 <- 4
sigma_3 <- 0
corr_12 <- 0.4
corr_13 <- 0
corr_23 <- 0

n <- 1000       # Число наблюдений
iter_all <- 50  # Число итераций
num_iter <- 1

# Генерация данных
h <- data_generate(n, name_exp, num_iter, li_type, error_type, 
                   sigma_1, sigma_2, sigma_3, corr_12, corr_13, corr_23)

# Оценивание модели байесовским методом
h_bs <- h

h_bs$transport <- as.factor(h_bs$transport)
h_bs <- one_hot(as.data.table(h_bs))

# Зависимая переменная -- 3 дамми
y <- cbind(h_bs$transport_Car,  h_bs$transport_Public,  h_bs$transport_Taxi)

X <- cbind(1,  h_bs$income,  h_bs$health,  h_bs$age)

data <- list(n = dim(X)[1],
             k = dim(X)[2],
             x = X,
             y = y)

model_normal <- stan(file = "C:\\Users\\S\\Documents\\Education\\Lina\\Thesis\\multinomial_probit_2_2_debug.stan",
                     data = data,               
                     chains = 1,                 # количество выборок из апостериорного распределения
                     iter = 2000)                # удвоенный объем выборки из апостериорного распределения
posterior_normal <- extract(model_normal)
