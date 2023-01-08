library(glue)

data_generate <- function(n=1000,
                          name_exp=1,
                          num_iter=1,
                          li_type="linear",
                          error_type="normal",
                          sigma_1=1,
                          sigma_2=3,
                          sigma_3=4,
                          corr_12=0.4,
                          corr_13=-0.6,
                          corr_23=0.3) {
  
  h <- data.frame(income = exp(rnorm(n, 10, 0.7)),           # показатель дохода
                  health = pmin(rpois(n, 3), 5),             # показатель здоровья
                  age = round(runif(n, 20, 100)))            # показатель возраста
  
  if (error_type=="normal") {
    sigma <- matrix(nrow=3,
                    data=c(sigma_1, corr_12, corr_13, 
                           corr_12, sigma_2, corr_23, 
                           corr_13, corr_23, sigma_3))
    eps <- rmvnorm(n, mean = rep(0, 3), sigma = sigma)       # случайные ошибки
  }
  
  # Истинные коэффициенты, линейные индексы и латентные переменные
  beta_Car <- c(0.1, 0.000025, 0.3, 0.01)
  beta_Taxi <- c(0.2, 0.000015, 0.2, 0.015)
  beta_Public <- c(3, -0.00002, 0.5, -0.02)
  
  if (li_type=="linear") {
    y_li_Car <- beta_Car[1] +                                  # линейный индекс Машины
                h$income * beta_Car[2] +
                h$health * beta_Car[3] +
                h$age * beta_Car[4]
    
    y_li_Taxi <- beta_Taxi[1] +                                # линейный индекс Такси
                 h$income * beta_Taxi[2] +
                 h$health * beta_Taxi[3] +
                 h$age * beta_Taxi[4]
    
    y_li_Public <- beta_Public[1] +                            # линейный индекс Общ. транспорта
                   h$income * beta_Public[2] +                           
                   h$health * beta_Public[3] +
                   h$age * beta_Public[4]
  }
  
  y_star_Car <- y_li_Car + eps[, 1]                          # латентная переменная Машины
  y_star_Taxi <- y_li_Taxi + eps[, 2]                        # латентная переменная Такси
  y_star_Public <- y_li_Public + eps[, 3]                    # латентная переменная Общ. транспорта
  
  # Зависимые переменные
  h$transport[(y_star_Car >= y_star_Taxi) &                  # те, кто выбрал Машину
              (y_star_Car >= y_star_Public)] <- "Car"
  h$transport[(y_star_Taxi >= y_star_Car) & 
              (y_star_Taxi >= y_star_Public)] <- "Taxi"      # те, кто выбрал Такси
  h$transport[(y_star_Public >= y_star_Car) & 
              (y_star_Public >= y_star_Taxi)] <- "Public"    # те, кто выбрал Общ. транспорт
  
  colnames(eps) <- c("eps_Car", "eps_Taxi", "eps_Public")
  full_h <- cbind(h, eps, y_star_Car, y_star_Taxi, y_star_Public)

  directory <- glue("results\\exp_{name_exp}\\n_{n}\\")
  
  dir.create(directory)
  write.csv(full_h, glue("{directory}\\iter_{num_iter}_data.csv"), row.names=FALSE)
  
  return(h)
}
