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
  int<lower=0> k;                      // число предикторов для всех уравнений
  matrix[n, k] x;                      // выборка для всех уравнений
  vector[n] y;                         // значение таргета
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
  matrix[2, 2] identity_matrix;

  beta_3 = rep_vector(0, k);
  identity_matrix = add_diag(identity_matrix, 1);
  sigma_1 = 1;
  
  mu_transformed_1[, 1] = (x * (beta_2-beta_1));
  mu_transformed_1[, 2] = (x * (beta_3-beta_1));
  
  mu_transformed_1[, 1] = (x * (beta_1-beta_2));
  mu_transformed_1[, 2] = (x * (beta_3-beta_2));
  
  mu_transformed_1[, 1] = (x * (beta_1-beta_3));
  mu_transformed_1[, 2] = (x * (beta_2-beta_3));
  
  transformation_matrix_1[1, 1] = 1;
  transformation_matrix_1[1, 2] = -1;
  transformation_matrix_1[1, 3] = 0;
  transformation_matrix_1[2, 1] = 1;
  transformation_matrix_1[2, 2] = 0;
  transformation_matrix_1[2, 3] = -1;
  
  transformation_matrix_2[1, 1] = -1;
  transformation_matrix_2[1, 2] = 1;
  transformation_matrix_2[1, 3] = 0;
  transformation_matrix_2[2, 1] = 0;
  transformation_matrix_2[2, 2] = 1;
  transformation_matrix_2[2, 3] = -1;
  
  transformation_matrix_3[1, 1] = -1;
  transformation_matrix_3[1, 2] = 0;
  transformation_matrix_3[1, 3] = 1;
  transformation_matrix_3[2, 1] = 0;
  transformation_matrix_3[2, 2] = -1;
  transformation_matrix_3[2, 3] = 1;
  
  Sigma[1, 1] = 1;
  Sigma[1, 2] = rho_12 * sigma_2;
  Sigma[1, 3] = 0;
  Sigma[2, 1] = rho_12 * sigma_2;
  Sigma[2, 2] = sigma_2 ^ 2;
  Sigma[2, 3] = 0;
  Sigma[3, 1] = 0;
  Sigma[3, 2] = 0;
  Sigma[3, 3] = 0;
  
  Sigma_1 = transformation_matrix_1 * Sigma * transformation_matrix_1';
  Sigma_2 = transformation_matrix_2 * Sigma * transformation_matrix_2';
  Sigma_3 = transformation_matrix_3 * Sigma * transformation_matrix_3';
}

// Описываем модель
model {
  // priors
  rho_12 ~ uniform(-1, 1);
  
  // likelihood
  for (i in 1:n){
    
    if (y[i] == 1) {
        y[i] ~ biprobit(mu_transformed_1[i, 1], mu_transformed_1[i, 2], 0);
    }
    
    else if (y[i] == 2) {
        y[i] ~ biprobit(mu_transformed_2[i, 1], mu_transformed_2[i, 2], 0);
    } 
    
    else if (y[i] == 3) {
        y[i] ~ biprobit(mu_transformed_3[i, 1], mu_transformed_3[i, 2], 0);
    } 
  }
}