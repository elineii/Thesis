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
  int<lower=0> k_1;                    // число предикторов для уравнений отбора
  int<lower=0> k_2;                    // число предикторов для уравнения интенсивности
  matrix[n, k_1] w;                    // выборка для уравнений отбора
  matrix[n, k_2] x;                    // выборка для уравнения интенсивности
  matrix[n, 1] y;                      // значение таргета уравнения интенсивности
}

// Указываем параметры
parameters {
  vector[k_1] gamma_1;                 // вектор коэффициентов первого уравнения отбора
  vector[k_1] gamma_2;                 // вектор коэффициентов второго уравнения отбора
  vector[k_2] beta;                    // вектор коэффициентов первого уравнения интенсивности
  real<lower=0> sigma_2;               // стандартное отклонение ошибки е_2
  real<lower=0> sigma_u;               // стандартное отклонение ошибки е_u (в уравнении интенсивности)
  real<lower=-1, upper=1> rho_12;      // корреляция между ошибками е_1 и е_2
  real<lower=-1, upper=1> rho_1u;      // корреляция между ошибками е_1 и u
  real<lower=-1, upper=1> rho_2u;      // корреляция между ошибками е_2 и u
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
  vector[3] initial_mu_1;
  vector[1] initial_mu_2;
  matrix[3, 3] initial_sigma_11;
  matrix[3, 1] initial_sigma_12;
  matrix[1, 3] initial_sigma_21;
  real initial_sigma_22;
  vector[3] conditional_mu;
  matrix[3, 3] conditional_sigma;

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
  
  initial_mu_1 = [0, 0, 0];
  initial_mu_2 = [0];
  initial_sigma_11 = Sigma;
  initial_sigma_12 = [rho_1u * sigma_u, rho_2u * sigma_2 * sigma_u];
  initial_sigma_21 = [rho_1u * sigma_u, sigma_2u * sigma_2 * sigma_u, 0];
  initial_sigma_22 = [sigma_u ^ 2];
}

// Описываем модель

model {
  real log_likelihood_sum = 0;
  vector[3] conditional_mu;
  matrix[3, 3] conditional_sigma;  
  
  // priors
  rho_12 ~ normal(0.4, 0.1);

  for (i in 1:n){
    if (y[i] == (-1)) {
      log_likelihood_sum = log_likelihood_sum + log(
        binormal_cdf(
          mu_transformed_2[i, 1], 
          mu_transformed_2[i, 2], 
          Sigma_2[1,1], 
          Sigma_2[2,2], 
          Sigma_2[1,2]
          )
        ) + log(
          binormal_cdf(
            mu_transformed_3[i, 1], 
            mu_transformed_3[i, 2], 
            Sigma_3[1,1], 
            Sigma_3[2,2], 
            Sigma_3[1,2]
            )
          );
    }
    
    if (y[i] != (-1)) {
      conditional_mu = initial_mu_1 + initial_sigma_12 * (1 / initial_sigma_22) * (y[i] - x[i] * beta);
      conditional_sigma = initial_sigma_11 - initial_sigma_12 * (1 / initial_sigma_22) * initial_sigma_21;
      conditional_mu_transformed[, 1] = x * (beta_2-beta_1) - (conditional_mu[1] - conditional_mu[2]);
      conditional_mu_transformed[, 2] = x * (beta_3-beta_1) - (conditional_mu[1] - conditional_mu[3]);
      conditional_sigma_transformed = transformation_matrix_1 * conditional_sigma * transformation_matrix_1';
      
      log_likelihood_sum = log_likelihood_sum + log(
        binormal_cdf(
          conditional_mu_transformed[i, 1], 
          conditional_mu_transformed[i, 2], 
          conditional_sigma_transformed[1,1], 
          conditional_sigma_transformed[2,2], 
          conditional_sigma_transformed[1,2]
          )
        );
    }
  }
}
