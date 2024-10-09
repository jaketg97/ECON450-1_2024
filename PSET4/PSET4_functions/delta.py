import numpy as np
from jax import jit
import jaxopt as jaxopt
from numba import jit
from PSET4_functions.shares import shares_m

###############################
## MAKING DELTA FUNCTIONS #####
###############################

@jit
def logit_delta(s_wide):
    s = s_wide 
    num_js = s.shape[1]
    deltas = np.empty((s.shape[0], num_js))
    for j in range(num_js):
        s_j = s[: , j]
        s_O = np.ones((1, len(s_j))) - np.sum(s, axis = 1)
        deltas[:, j] = np.log(s_j) - np.log(s_O)

    return deltas.flatten()

@jit
def contraction_map_m(delta_0_m, theta_2, real_sm_logged, p_m, random_vs, tol = 1e-13, max_iterations=100000): 
    
    prev_delta = delta_0_m 
    for i in range(max_iterations):

        # get estimated shares given delta
        est_shares = shares_m(theta_2, prev_delta, p_m, random_vs)
        
        # log those estimated shares (in a weird way to make jit compilation work)
        est_shares_logged = np.zeros(len(est_shares)) 
        for j in range(len(est_shares)):
            est_shares_logged[j] = np.log(est_shares[j])
        
        # contraction mapping itself
        delta = prev_delta + real_sm_logged - est_shares_logged
        
        # check tolerance
        current_tol = np.linalg.norm(delta-prev_delta, ord = np.inf)
        if current_tol < tol: 
            break
        
        # rinse and repeat
        prev_delta = delta
        
    return delta, i, current_tol

@jit
def inner_loop(theta_2, delta_0, s_wide, p_wide, random_vs): 
    s = s_wide 
    p = p_wide
    num_ms = s.shape[0]
    num_js = s.shape[1]

    deltas = np.empty((num_ms, num_js))
    loops = np.empty((num_ms, num_js))
    tols = np.empty((num_ms, num_js))

    for m in range(num_ms):

        p_m = p[m] 

        delta_0_m = delta_0[m]
        
        real_sm = s[m]
        real_s_logged = np.zeros(len(real_sm))
        for j in range(len(real_sm)):
            real_s_logged[j] = np.log(real_sm[j])
        
        delta, num_loops, current_tol = contraction_map_m(delta_0_m, theta_2, real_s_logged, p_m, random_vs)
        
        deltas[m] = delta
        loops[m] = num_loops
        tols[m] = current_tol 
    
    return deltas.flatten()