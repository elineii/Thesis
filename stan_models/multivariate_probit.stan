// Bivariate normal cumulative distribution function
// https://mc-stan.org/docs/2_28/stan-users-guide/examples.html#bivariate-normal-cumulative-distribution-function

// https://discourse.mc-stan.org/t/bivariate-probit-in-stan/2025/7

functions {
  real binormal_cdf(real z1, real z2, real rho) {
    if (z1 != 0 || z2 != 0) {
      real denom = fabs(rho) < 1.0 ? sqrt((1 + rho) * (1 - rho)) 
                                   : not_a_number();
      real a1 = (z2 / z1 - rho) / denom;
      real a2 = (z1 / z2 - rho) / denom;
      real product = z1 * z2;
      real delta = product < 0 || (product == 0 && (z1 + z2) < 0);
      return 0.5 * (Phi(z1) + Phi(z2) - delta)
                   - owens_t(z1, a1) - owens_t(z2, a2);
    }
    return 0.25 + asin(rho) / (2 * pi());
  }
  
  real biprobit_lpdf(row_vector Y, real mu1, real mu2, real rho) {
    real q1;
    real q2;
    real w1;
    real w2;
    real rho1;
    
    // from Greene's econometrics book
    q1 = 2*Y[1] - 1.0;
    q2 = 2*Y[2] - 1.0;
    
    w1 = q1*mu1;
    w2 = q2*mu2;
    
    rho1 = q1*q2*rho;
    return log(binormal_cdf(w1, w2, rho1));
  }
}

// Описываем входные данные
data {
  int<lower=0> n;                      // число наблюдений
  int<lower=0> k_1;                    // число предикторов для обеих моделей
  int<lower=0> k_2;
  matrix[n, k_1] x_1;                  // выборка для первой модели
  matrix[n, k_2] x_2;                  // выборка для второй модели
  matrix[n, 2] y;                      // значение регрессии
}

// Указываем параметры
parameters {
  vector[k_1] beta_1;                   // вектор коэффициентов 
  vector[k_2] beta_2;
  real<lower=-1, upper=1> rho;
}

transformed parameters {
  matrix[n, 2] mu;
  mu[, 1] = (x_1 * beta_1);
  mu[, 2] = (x_2 * beta_2);
}

// Описываем модель
model {
  // priors
  beta_1 ~ normal(0, 10);
  beta_2 ~ normal(0, 10);
  // likelihood
  for (i in 1:n){
    y[i, ] ~ biprobit(mu[i, 1], mu[i, 2], rho);
  }
}