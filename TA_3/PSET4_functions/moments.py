import numpy as np
from jax import jit
import jaxopt as jaxopt
from numba import jit

def hausman_instruments(p_wide): 
    hausman_instruments = np.zeros((p_wide.shape[0], p_wide.shape[1]))
    for j in range(p_wide.shape[1]):
        for m in range(p_wide.shape[0]):
            hausman_instruments[m, j] = np.delete(p_wide[:, j], m, axis = 0).mean() 
    return hausman_instruments.flatten()

@jit
def blp_instruments(x_3d):
    num_markets = x_3d.shape[0]
    num_prods = x_3d.shape[1]
    x_3d = np.ascontiguousarray(x_3d)
    own_chars = x_3d.reshape(num_markets*num_prods, 3)

    sum_chars = np.sum(x_3d, axis = 1)
    blp_rival = np.empty_like(x_3d)
    for i in range(x_3d.shape[0]):
        for j in range(x_3d.shape[1]):
            blp_rival[i, j] = sum_chars[i] - x_3d[i, j]

    blp_rival = blp_rival.reshape(num_markets*num_prods, 3)
    blp_rival = blp_rival[:, 1:]
    return np.concatenate((own_chars, blp_rival), axis = 1)

###############################
## MOMENT FUNCTIONS ###########
###############################

@jit
def g(delta, prices_wide, x_long, x_3d, z):
    num_m = prices_wide.shape[0]
    num_j = prices_wide.shape[1]
    prices = prices_wide
    x = np.concatenate((x_long, prices.flatten().reshape(-1,1)), axis = 1)
    p_z = z @ np.linalg.solve(z.T @ z, np.eye(z.shape[1])) @ z.T
    x_pz_x_inv = np.linalg.solve(x.T @ p_z @ x, np.eye(x.shape[1]))
    A = np.eye(len(delta)) - x @ x_pz_x_inv @ x.T @ p_z
    return z.T @ A @ delta

@jit
def g_ddelta(delta, prices_wide, x_long, x_3d, z):
    num_m = prices_wide.shape[0]
    num_j = prices_wide.shape[1]
    prices = prices_wide
    x = np.concatenate((x_long, prices.flatten().reshape(-1,1)), axis = 1)
    p_z = z @ np.linalg.solve(z.T @ z, np.eye(z.shape[1])) @ z.T
    x_pz_x_inv = np.linalg.solve(x.T @ p_z @ x, np.eye(x.shape[1]))
    A = np.eye(len(delta)) - x @ x_pz_x_inv @ x.T @ p_z
    return z.T @ A

# FOR SE ESTIMATION 
def g_jm(delta, prices_wide, x_long, z):
    num_m = prices_wide.shape[0]
    num_j = prices_wide.shape[1]
    prices = prices_wide
    x = np.concatenate((x_long, prices.flatten().reshape(-1,1)), axis = 1)
    p_z = z @ np.linalg.solve(z.T @ z, np.eye(z.shape[1])) @ z.T
    x_pz_x_inv = np.linalg.solve(x.T @ p_z @ x, np.eye(x.shape[1]))
    A = np.eye(len(delta)) - x @ x_pz_x_inv @ x.T @ p_z
    return (A @ delta).reshape(-1,1) * z