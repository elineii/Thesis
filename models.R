library("switchSelection")

options(scipen = 999)
set.seed(123)

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
