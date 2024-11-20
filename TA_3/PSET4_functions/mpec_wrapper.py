# Pre-question, loading libraries and data

from statsmodels.compat.python import lrange

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import optimize, stats
import scipy.integrate as integrate
from scipy.stats import lognorm
from scipy.io import loadmat

from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
                                    LikelihoodModel, LikelihoodModelResults)
from statsmodels.regression.linear_model import (OLS, RegressionResults,
                                                 RegressionResultsWrapper)
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d
from statsmodels.sandbox.regression.gmm import IV2SLS

from sklearn.preprocessing import PolynomialFeatures

import jax.numpy as jnp
from jax import grad, jit, vmap, jacfwd, jacrev
import jaxopt as jaxopt
import re

import sympy as sp
from tabulate import tabulate
import matplotlib.pyplot as plt

from numba import jit
from PSET4_functions.shares import *
from PSET4_functions.moments import *


###############################
## MPEC WRAPPER FUNCTIONS #####
###############################

@jit 
def mpec_objective(params, num_instruments):
    data_len = params.shape[0] - num_instruments - 1
    theta_2 = params[0]
    delta = params[1: data_len + 1]
    eta = params[data_len + 1: data_len + 1 + num_instruments]
    return eta.T @ np.eye(num_instruments) @ eta

@jit 
def mpec_gradient(params, num_instruments):
    data_len = params.shape[0] - num_instruments - 1
    theta_2 = params[0]
    delta = params[1: data_len + 1]
    eta = params[data_len + 1: data_len + 1 + num_instruments]
    d_dtheta2 = np.zeros(1)
    d_ddelta = np.zeros((len(delta), ))
    d_deta = 2 * eta 
    return np.concatenate((d_dtheta2, d_ddelta, d_deta))

@jit 
def mpec_constraints(params, z, prices_wide, x_long, x_3d, shares_long, random_vs):
    data_len = x_long.shape[0]
    prices = prices_wide
    theta_2 = params[0]
    delta = params[1: data_len + 1]
    num_instruments = z.shape[1]
    eta = params[data_len + 1: data_len + 1 + num_instruments]
    term1 = np.concatenate((shares(theta_2, delta, prices, random_vs), g(delta, prices, x_long, x_3d, z)))
    term2 = np.concatenate((shares_long.flatten(), eta))
    return term1 - term2

@jit 
def mpec_jacobian(params, z, prices_wide, x_long, x_3d, shares_long, random_vs):
    data_len = x_long.shape[0]
    num_instruments = z.shape[1]
    prices = prices_wide
    theta_2 = params[0]
    delta = params[1: data_len + 1]
    eta = params[data_len + 1: data_len + 1 + num_instruments]
    ds_dtheta2 = shares_dtheta(theta_2, delta, prices, random_vs)
    ds_ddelta = shares_ddelta(theta_2, delta, prices, random_vs)
    dg_ddelta = g_ddelta(delta, prices, x_long, x_3d, z)
    negative_i_eta = - np.eye(dg_ddelta.shape[0])
    top_row = np.hstack((ds_dtheta2.reshape(-1,1), ds_ddelta, np.zeros((data_len,num_instruments))))
    bottom_row = np.hstack((np.zeros((dg_ddelta.shape[0],1)), dg_ddelta, negative_i_eta))
    return np.concatenate((top_row, bottom_row))

@jit
def ivregress_fast(y, x, z):
    p_z = z @ np.linalg.solve(z.T @ z, np.eye(z.shape[1])) @ z.T
    x_pz_x_inv = np.linalg.solve(x.T @ p_z @ x, np.eye(x.shape[1]))
    return x_pz_x_inv @ x.T @ p_z @ y

@jit
def beta_alpha(delta, z, prices_long, x_long, x_3d):
    prices = prices_long
    x_tilde = np.concatenate((x_long, prices), axis = 1)
    coefs = ivregress_fast(delta, x_tilde, z)
    return (coefs[0], coefs[1], coefs[2]), coefs[3]

def full_mpec_wrapper(theta_2, dat, z_data, random_vs, num_markets, num_prods): 
    
    num_instruments = z_data.shape[1]
    prices_data_long = dat[["pjm"]].to_numpy()
    prices_data_wide = prices_data_long.reshape(num_markets, num_prods)

    shares_data_long = dat[['sjm']].to_numpy()
    shares_data_wide = shares_data_long.reshape(num_markets,num_prods)

    x_data_long = dat[['X1jm', 'X2jm', 'X3jm']].to_numpy()
    x_data_3d = x_data_long.reshape(num_markets, num_prods, 3)

    delta_0 = dat[['delta_0']].to_numpy().flatten() 
    eta_0 = np.zeros(num_instruments)

    params_0 = np.concatenate((theta_2, delta_0, eta_0))
    given_args = (z_data, prices_data_wide, x_data_long, x_data_3d, shares_data_long, random_vs)

    mpec_cons = {'type': 'eq', 'fun': mpec_constraints, 'args': given_args, 'jac': mpec_jacobian}

    solution_bounds = [(.1, None)] + [(None, None)] * (num_markets * num_prods + num_instruments)
    mpec_soln = optimize.minimize(mpec_objective, params_0, args = num_instruments, jac = mpec_gradient, method = 'SLSQP', bounds = solution_bounds, constraints = mpec_cons, options = {'maxiter':100})
    mpec_soln_values = mpec_soln.x
    sigma_alpha = mpec_soln_values[0]
    delta_hat = mpec_soln_values[1:(num_prods*num_markets + 1)]
    beta, alpha = beta_alpha(delta_hat, z_data, prices_data_long, x_data_long, x_data_3d)
    return beta, alpha, sigma_alpha, delta_hat

def boot(theta_2, data, z_data, random_vs, num_markets, num_prods, reps):
    coefs = np.zeros((reps, 5))
    for i in range(reps):
        temp = data.groupby("m").sample(replace = True, random_state=i, frac = 1)
        beta, alpha, sigma_alpha, delta_hat = full_mpec_wrapper(theta_2, temp, z_data, random_vs, num_markets, num_prods)
        beta_1 = beta[0]
        beta_2 = beta[1]
        beta_3 = beta[2]
        coefs[i] = np.array([beta_1, beta_2, beta_3, alpha, sigma_alpha])
    return coefs

def standard_errors(theta_2, delta_long, z, prices_wide, x_long, x_3d, random_vs):
    
    # From Tomas's notes...
    ds_ddelta_inv = np.linalg.solve(shares_ddelta(theta_2, delta_long, prices_wide, random_vs), np.eye(delta_long.shape[0]))
    ds_dtheta = shares_dtheta(theta_2, delta_long, prices_wide, random_vs) 
    G = (z.T @ (ds_ddelta_inv @ ds_dtheta.reshape(-1,1)))
    B = (g_jm(delta_long, prices_wide, x_long, z).T @ g_jm(delta_long, prices_wide, x_long, z))
    var_theta2 = (1/(G.T @ G) * G.T @ B @ G * 1/(G.T @ G))

    # Now apply delta method
    x = np.concatenate((x_long, prices_wide.flatten().reshape(-1,1)), axis = 1)
    p_z = z @ np.linalg.solve(z.T @ z, np.eye(z.shape[1])) @ z.T
    x_pz_x_inv = np.linalg.solve(x.T @ p_z @ x, np.eye(x.shape[1]))
    coef_matrix = x_pz_x_inv @ x.T @ p_z
    var_beta_alpha = var_theta2 * np.square((coef_matrix @ (ds_ddelta_inv @ ds_dtheta)))

    return np.sqrt(var_beta_alpha).flatten(), np.sqrt(var_theta2).flatten()