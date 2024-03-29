library("switchSelection")
library("rstan")

MLE <- function(df_path, res_path) { 
  res <- data.frame(beta_0=NA, beta_1=NA, beta_3=NA)
  
  df <- read.csv(df_path)
  df <- data.frame(w0=df$w0, w1=df$w1, w2=df$w2, z=df$z,
                   x0=df$x0, x1=df$x1, x2=df$x2, y=df$y)
  df$y[is.na(df$y)] <- Inf
  
  tryCatch({
    model <- mnprobit(z ~ w1 + w2,
                      y ~ x1 + x2,
                      regimes = c(0, -1, -1),
                      data = df, cov_type = "gop")
    res <- model$tbl$coef2[[1]][1:3]
    res <- data.frame(beta_0=res[1], beta_1=res[2], beta_3=res[3])
  }, error = function(e) {})
  write.csv(res, res_path, row.names=FALSE)
}

Bayes <- function(df_path, res_path) { 
  df <- read.csv(df_path)
  df <- data.frame(w0=df$w0, w1=df$w1, w2=df$w2, z=df$z,
                   x0=df$x0, x1=df$x1, x2=df$x2, y=df$y)
  df$y[is.na(df$y)] <- Inf
  
  tryCatch({
    # Fit ML model
    model <- mnprobit(z ~ w1 + w2,
                      y ~ x1 + x2,
                      regimes = c(0, -1, -1),
                      data = df, cov_type = "gop")

    # TODO: ВЫТАЩИТЬ МАТ ОЖИДАНИЕ И СТАНДАРТНОЕ ОТКЛОНЕНИЕ ГАММ И БЕТЫ
    res <- model$tbl$coef2[[1]][1:3]
    res <- data.frame(beta_0=res[1], beta_1=res[2], beta_3=res[3])
    beta_exp_ml <- 
    beta_exp_sd <-
    gamma_1_exp_ml <- 
    gamma_1_sd_ml <- 
    gamma_2_exp_ml <- 
    gamma_2_sd_ml <-

    # Fit bayes model
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
                  data = data,                
                  chains = 1,                 
                  iter = 2000)
    res <- extract(model)
  }, error = function(e) {})
  saveRDS(res, file=res_path)
}
