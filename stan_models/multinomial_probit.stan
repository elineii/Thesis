// Bivariate normal cumulative distribution function
// https://mc-stan.org/docs/2_28/stan-users-guide/examples.html#bivariate-normal-cumulative-distribution-function

// https://discourse.mc-stan.org/t/bivariate-probit-in-stan/2025/7

functions {
  real binormal_cdf(real z1, real z2, real var1, real var2, real cov) {
    real z1_st;
    real z2_st;
    real rho;
    z1_st = z1 / (var1 ^ 0.5);
    z2_st = z2 / (var2 ^ 0.5);
    rho = cov / (var1 ^ 0.5 * var2 ^ 0.5);
    if (z1_st != 0 || z2_st != 0) {
      real denom = fabs(rho) < 1.0 ? sqrt((1 + rho) * (1 - rho)) 
                                   : not_a_number();
      real a1 = (z2_st / z1_st - rho) / denom;
      real a2 = (z1_st / z2_st - rho) / denom;
      real product = z1_st * z2_st;
      real delta = product < 0 || (product == 0 && (z1_st + z2_st) < 0);
                   
      return fmax(1.70247e-69, 0.5 * (Phi(z1_st) + Phi(z2_st) - delta)
                   - owens_t(z1_st, a1) - owens_t(z2_st, a2));
    }
    return fmax(1.70247e-69, 0.25 + asin(rho) / (2 * pi()));
  }
}

// Описываем входные данные
data {
  int<lower=0> n;                      // число наблюдений
  int<lower=0> k;                      // число предикторов для всех уравнений
  matrix[n, k] x;                      // выборка для всех уравнений
  matrix[n, 3] y;                      // значение таргета
}

// Указываем параметры
parameters {
  vector[k] beta_1;                    // вектор коэффициентов первого уравнения 
  vector[k] beta_2;                    // вектор коэффициентов второго уравнения
  real<lower=0> sigma_2;               // стандартное отклонение ошибки е_2
  real<lower=-1, upper=1> rho_12;      // корреляция между ошибками е_1 и е_2
}

transformed parameters {
  vector[k] beta_3;
  real sigma_1;
  matrix[n, 2] mu_transformed_1;
  matrix[n, 2] mu_transformed_2;
  matrix[n, 2] mu_transformed_3;
  matrix[2, 3] transformation_matrix_1;
  matrix[2, 3] transformation_matrix_2;
  matrix[2, 3] transformation_matrix_3;
  matrix[3, 3] Sigma;
  matrix[2, 2] Sigma_1;
  matrix[2, 2] Sigma_2;
  matrix[2, 2] Sigma_3;

  beta_3 = rep_vector(0, k);
  sigma_1 = 1;
  
  mu_transformed_1[, 1] = (x * (beta_2-beta_1));
  mu_transformed_1[, 2] = (x * (beta_3-beta_1));
  
  mu_transformed_2[, 1] = (x * (beta_1-beta_2));
  mu_transformed_2[, 2] = (x * (beta_3-beta_2));
  
  mu_transformed_3[, 1] = (x * (beta_1-beta_3));
  mu_transformed_3[, 2] = (x * (beta_2-beta_3));
  
  transformation_matrix_1 = [
    [1, -1, 0],
    [1, 0, -1]
  ];
  
  transformation_matrix_2 = [
    [-1, 1, 0],
    [0, 1, -1]
  ];
  
  transformation_matrix_3 = [
    [-1, 0, 1],
    [0, -1, 1]
  ];
  
  Sigma = [
    [1, rho_12 * sigma_2, 0],
    [rho_12 * sigma_2, sigma_2 ^ 2, 0],
    [0, 0, 0]
  ];
  
  Sigma_1 = transformation_matrix_1 * Sigma * transformation_matrix_1';
  Sigma_2 = transformation_matrix_2 * Sigma * transformation_matrix_2';
  Sigma_3 = transformation_matrix_3 * Sigma * transformation_matrix_3';
}

// Описываем модель

model {
  real log_likelihood_sum = 0;
  
  // priors
  rho_12 ~ normal(0.4, 0.1);

1 0 0
0 1 0
0 0 1

  for (i in 1:n){
    if ((y[i, 1] == 1) && (y[i, 2] == 0) && (y[i, 3] == 0)) {
      log_likelihood_sum = log_likelihood_sum + log(
        binormal_cdf(
          mu_transformed_1[i, 1], 
          mu_transformed_1[i, 2], 
          Sigma_1[1,1], 
          Sigma_1[2,2], 
          Sigma_1[1,2]
          )
        );
    }
    
    if ((y[i, 1] == 0) && (y[i, 2] == 1) && (y[i, 3] == 0)) {
      log_likelihood_sum = log_likelihood_sum + log(
        binormal_cdf(
          mu_transformed_2[i, 1], 
          mu_transformed_2[i, 2], 
          Sigma_2[1,1], 
          Sigma_2[2,2], 
          Sigma_2[1,2]
          )
        );
    }
    
    if ((y[i, 1] == 0) && (y[i, 2] == 0) && (y[i, 3] == 1)) {
      log_likelihood_sum = log_likelihood_sum + log(
        binormal_cdf(
          mu_transformed_3[i, 1], 
          mu_transformed_3[i, 2], 
          Sigma_3[1,1], 
          Sigma_3[2,2], 
          Sigma_3[1,2]
          )
        );
    }
  }
}
