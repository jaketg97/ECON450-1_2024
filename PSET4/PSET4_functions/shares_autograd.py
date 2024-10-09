import numpy as np
from jax import jit, jacfwd
import jax.numpy as jnp
import jaxopt as jaxopt

######################
## SHARES FUNCTIONS ##
######################

def shares_im(theta_2, delta_m, p_m, v):
    sigma_alpha = theta_2 
    normalize_share = 0
    numerators = jnp.exp(delta_m - (sigma_alpha * v) * p_m - normalize_share)
    s_jms = jnp.empty(len(delta_m), dtype=np.float32) 
    for i in range(len(numerators)):
       s_jms = s_jms.at[i].set(numerators[i]/(1 * jnp.exp(-normalize_share) + jnp.sum(numerators)))
    return s_jms

shares_im_jit = jit(shares_im)
shares_im_ds_ddelta = jacfwd(shares_im, argnums = (1))
shares_im_ds_ddelta_jit = jit(shares_im_ds_ddelta)

def shares_m(theta_2, delta_m, p_m, random_vs):
    shares_m = jnp.empty((len(random_vs), len(delta_m)))
    i = 0 
    for v in random_vs: 
        shares_m = shares_m.at[i].set(shares_im(theta_2, delta_m, p_m, v))
        i = i+1
    return [shares_m[:,j].mean() for j in range(len(delta_m))]

def shares_m_ds_ddelta(theta_2, delta_m, p_m, random_vs):
    shares_m = jnp.zeros((len(delta_m), len(delta_m)))
    i = 0 
    for v in random_vs: 
        shares_m = shares_m + shares_im_ds_ddelta(theta_2, delta_m, p_m, v)
        i = i+1
    return shares_m * 1/len(random_vs)

shares_m_jit = jit(shares_m)
shares_m_ds_ddelta_jit = jit(shares_m_ds_ddelta)

def shares(theta_2, delta_long, p_wide, random_vs):
    p = p_wide
    num_markets = p_wide.shape[0]
    num_prods = p_wide.shape[1]
    delta = delta_long.reshape(num_markets, num_prods)
    num_ms = p.shape[0]
    num_js = p.shape[1]
    shares = jnp.empty((num_ms, num_js))
    for m in range(num_ms):
        p_m = p[m]
        delta_m = delta[m]
        shares =shares.at[m].set(shares_m(theta_2, delta_m, p_m, random_vs))
    
    return shares.flatten() # I convert back to 1-d since that works better for MPEC constraints

def shares_ds_ddelta(theta_2, delta_long, p_wide, random_vs):
    p = p_wide
    num_markets = p_wide.shape[0]
    num_prods = p_wide.shape[1]
    delta = delta_long.reshape(num_markets, num_prods)
    num_ms = p.shape[0]
    num_js = p.shape[1]
    shares = jnp.empty((num_ms*num_js, num_ms*num_js))
    for m in range(num_ms):
        p_m = p[m]
        delta_m = delta[m]
        shares =shares.at[m*num_js:(m+1)*num_js, m*num_js:(m+1)*num_js].set(shares_m_ds_ddelta_jit(theta_2, delta_m, p_m, random_vs))
    
    return shares.flatten() # I convert back to 1-d since that works better for MPEC constraints

shares_ds_ddelta_jit = jit(shares_ds_ddelta)
