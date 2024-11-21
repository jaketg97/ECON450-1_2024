import numpy as np
from jax import jit
import jaxopt as jaxopt
from numba import jit

######################
## SHARES FUNCTIONS ##
######################

@jit
def shares_im(theta_2, delta_m, p_m, v):
    sigma_alpha = theta_2 
    normalize_share = 0
    numerators = np.exp(delta_m - (sigma_alpha * v) * p_m - normalize_share)
    s_jms = np.empty(len(delta_m), dtype=np.float64) 
    for i in range(len(numerators)):
        s_jms[i] = numerators[i]/(1 * np.exp(-normalize_share) + np.sum(numerators))
    return s_jms  

@jit
def shares_m(theta_2, delta_m, p_m, random_vs):
    shares_m = np.empty((len(random_vs), len(delta_m)))
    i = 0 
    for v in random_vs: 
        shares_m[i] = shares_im(theta_2, delta_m, p_m, v)
        i = i+1
    return [shares_m[:,j].mean() for j in range(len(delta_m))]

@jit 
def shares(theta_2, delta_long, p_wide, random_vs):
    p = p_wide
    num_markets = p_wide.shape[0]
    num_prods = p_wide.shape[1]
    delta = delta_long.reshape(num_markets, num_prods)
    num_ms = p.shape[0]
    num_js = p.shape[1]
    shares = np.empty((num_ms, num_js))
    for m in range(num_ms):
        p_m = p[m]
        delta_m = delta[m]
        shares[m] = shares_m(theta_2, delta_m, p_m, random_vs)
    
    return shares.flatten() # I convert back to 1-d since that works better for MPEC constraints

##############################
## SHARES DELTA DERIVATIVES ##
##############################

@jit 
def shares_im_ddelta(theta_2, delta_m, p_m, v):
    s_ijm = shares_im(p_m, delta_m, theta_2, v)
    grad_matrix = s_ijm.reshape(-1,1)@-s_ijm.reshape(-1,1).T 
    s_ijm_diag = np.diag(s_ijm)
    grad_matrix = grad_matrix + s_ijm_diag 
    return grad_matrix

@jit 
def shares_m_ddelta(theta_2, delta_m, p_m, random_vs):
    shares_ddelta = np.zeros((len(delta_m), len(delta_m))) 
    i = 0 
    for v in random_vs: 
        shares_ddelta = shares_ddelta + shares_im_ddelta(theta_2, delta_m, p_m, v) #sum matrices
        i = i+1
    return shares_ddelta * 1/len(random_vs) #divide to get average matrix

@jit 
def shares_ddelta(theta_2, delta_long, p, random_vs):
    num_ms = p.shape[0]
    num_js = p.shape[1]
    delta = delta_long.reshape(num_ms, num_js)
    jacobian = np.zeros((num_js*num_ms, num_js*num_ms))
    index = 0
    for m in range(num_ms):
        p_m = p[m]
        delta_m = delta[m]
        block = shares_m_ddelta(theta_2, delta_m, p_m, random_vs)
        size = block.shape[0]
        jacobian[index : index + size, index : index + size] = block
        index += size 
 
    return jacobian

###############################
## SHARES THETA2 DERIVATIVES ##
###############################

@jit
def shares_im_dtheta(theta_2, delta_m, p_m, v):
    sigma_alpha = theta_2 
    normalize_share = 0
    numerators = np.exp(delta_m - (sigma_alpha * v) * p_m - normalize_share)
    d_numerators = -v * p_m * numerators 
    denominator = 1 * np.exp(-normalize_share) + np.sum(numerators)
    s_jms_deltas = np.empty(len(delta_m), dtype=np.float64) 
    for i in range(len(numerators)):
        s_jms_deltas[i] = (d_numerators[i] * denominator - numerators[i] *  np.sum(d_numerators))/(np.square(denominator))
    return s_jms_deltas  

@jit
def shares_m_dtheta(theta_2, delta_m, p_m, random_vs):
    shares_m_dtheta = np.empty((len(random_vs), len(delta_m)))
    i = 0 
    for v in random_vs: 
        shares_m_dtheta[i] = shares_im_dtheta(theta_2, delta_m, p_m, v)
        i = i+1
    return [shares_m_dtheta[:,j].mean() for j in range(len(delta_m))]

@jit 
def shares_dtheta(theta_2, delta_long, p, random_vs):
    num_ms = p.shape[0]
    num_js = p.shape[1]
    delta = delta_long.reshape(num_ms, num_js)
    shares_dtheta = np.empty((num_ms, num_js))
    for m in range(num_ms):
        p_m = p[m]
        delta_m = delta[m]
        shares_dtheta[m] = shares_m_dtheta(theta_2, delta_m, p_m, random_vs)
    
    return shares_dtheta.flatten() # I convert back to 1-d since that works better for MPEC constraints
