import numpy as np
import pandas as pd
import os

def get_simulation(
    n=1000, 
    rho=np.array([[1.0, 0.1, 0.45, 0.64 * np.sqrt(32.0)], 
                  [0.1, 1.0, -0.35, -0.24 * np.sqrt(32.0)], 
                  [0.45, -0.35, 1.0, 0.14 * np.sqrt(32.0)], 
                  [0.64 * np.sqrt(32.0), -0.24 * np.sqrt(32.0), 0.14 * np.sqrt(32.0), 32.0]]),
    r=0.9, 
    beta=np.array([1.0, 1.0, 1.0]),
    gamma=np.array([[3.3, 1.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0]])
    ):
    """
    Generate data for simulations on the basis of the article (Bourguignon et al, 2007):
    
    y_1* = beta_1 + beta_2 * x_1 + beta_3 * x_2 + eps_1
    z_1* = gamma_11 + gamma_12 * w_1 + gamma_13 * w_2 + u_1
    z_2* = gamma_21 + gamma_22 * w_1 + gamma_23 * w_2 + u_2
    z_3* = gamma_31 + gamma_32 * w_1 + gamma_33 * w_2 + u_3
    
    x_1 = r * w_1 + v_1
    x_2 = r * w_2 + v_2
            
    w_1, w_2 ~ N(0, 4)
    v_1, v_2 ~ N(0, 4 * sqrt(1 - r ** 2))
    
    Args:
        - n: sample size (target in the main equation will be visible only for about a third of the observations)
        - rho: covariance matrix of multivariate normal distribution of random errors
        - r: parameter used to generate regressors of the main equation
        - beta: coefficients of the main equation
        - gamma: coefficients of selection equations
        
    Note: for z != 1 values of y are not observed
    """
    errors = np.random.multivariate_normal(mean=np.zeros(4), cov=rho, size=n)
    u, eps = errors[:, :-1], errors[:, -1]
    v = np.random.normal(size=(n, 2), loc=0, scale=(4 * np.sqrt(1 - r ** 2)))
    
    W = np.random.normal(size=(n, 2), loc=0, scale=4)
    X = r * W + v
    W_w_const = np.hstack((np.ones((n, 1)), W))
    X_w_const = np.hstack((np.ones((n, 1)), X))
    
    Z_star = W_w_const @ gamma + u
    z = np.argmax(Z_star, axis=1) + 1
    y = X_w_const @ beta + eps
    
    data = np.hstack((W_w_const, X_w_const, Z_star, 
                      z.reshape((n, 1)), y.reshape((n, 1))))
    
    cols = ['w0', 'w1', 'w2','x0', 'x1', 'x2',
               'z_star1', 'z_star2', 'z_star3', 'z', 'y']
    df = pd.DataFrame(data, columns=cols)
    df['y_star'] = df['y']
    df.loc[df.z != 1, 'y'] = np.nan
    
    return df


def get_data_simulations(n: list=[50, 100, 200, 1000, 10000], n_sim=100, path="data/simulations"):
    """
    Generate multiple simulations of the same type.
    
    Args:
        - n: sample size (target in the main equation will be visible only for about a third of the observations)
        - n_sim: number of simulations
    """
    for cur_sim in range(n_sim):
        for cur_n in n:
            df = get_simulation(n=cur_n)
            
            if not os.path.exists(f'{path}/n_{cur_n}'):
                os.makedirs(f'{path}/n_{cur_n}')
            
            df.to_csv(f'{path}/n_{cur_n}/sim_{cur_sim}_n_{cur_n}.csv', index=False)
            
            
if __name__ == '__main__':
    np.random.seed(999)
    get_data_simulations(path="data/simulation_benchmark")
