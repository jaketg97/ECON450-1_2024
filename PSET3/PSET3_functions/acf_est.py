import numpy as np
import statsmodels.api as sm
import jax.numpy as jnp
import jaxopt as jaxopt
from PSET3_functions.misc import poly_2v, poly_3v, jnp_reg_predict
from scipy import optimize

def ACF_estimation(dat):

    ACF_data = dat.copy()

    # making lags we'll use later
    ACF_data["k_lag"] = ACF_data.groupby("firm_id")["k"].shift(1)
    ACF_data["l_lag"] = ACF_data.groupby("firm_id")["l"].shift(1)

    #####
    # ACF First Stage 
    #####

    ACF_data, phi_vars = poly_3v("l", "k", "m", ACF_data)
    constant = jnp.ones((ACF_data.shape[0], 1))
    phi_var_mat = jnp.array(ACF_data[phi_vars].to_numpy())
    # acf_first_stage = sm.OLS(ACF_data["y"], sm.add_constant(ACF_data[phi_vars])).fit()
    # acf_first_stage.resid
    X = jnp.hstack((constant, phi_var_mat))
    y = ACF_data["y"].copy()
    ACF_data["y_hat"] = jnp_reg_predict(y, X)
    ACF_data["phi_prediction"] = ACF_data["y_hat"] 
    ACF_data["phi_prediction_lag"] = ACF_data.groupby("firm_id")["phi_prediction"].shift(1)

    #####
    # ACF Second Stage 
    #####

    # dropping NAs based on lags
    ACF_data = ACF_data[ACF_data["k_lag"].notna()]
    ACF_data = ACF_data[ACF_data["l_lag"].notna()]
    ACF_data = ACF_data[ACF_data["phi_prediction_lag"].notna()]
        
        # index keeping, as we move to all numpy here so I can use the autograd
    ACF_data_colnames = list(ACF_data.columns)
    column_indices = {}
    for index, column_name in enumerate(ACF_data_colnames):
        column_indices[column_name] = index

        # GMM objective function (that's auto-differentiable, hence the _grad)
    def ACF_GMM_val_grad(params, data):
        beta_k, beta_l = params
        dat = data 
        omega = dat[:, column_indices["phi_prediction"]] - beta_k * dat[:, column_indices["k"]] - beta_l * dat[:, column_indices["l"]]
        omega_lag = dat[:, column_indices["phi_prediction_lag"]] - beta_k * dat[:, column_indices["k_lag"]] - beta_l * dat[:, column_indices["l_lag"]]
        g = omega_lag.reshape(-1, 1)
        g2 = jnp.power(omega_lag, 2).reshape(-1, 1)
        g3 = jnp.power(omega_lag, 3).reshape(-1, 1)
        g_func = jnp.hstack((jnp.ones((g.shape[0], 1)),g, g2, g3))
        # return omega, g_func
        omega_hat = jnp_reg_predict(omega, g_func)
        ksi = omega - omega_hat
        instruments = ["k", "l_lag"]
        moments = jnp.array([])
        for z in instruments:
            z_moment_col = dat[: , column_indices[z]] * ksi
            z_moment = z_moment_col.mean()
            moments = jnp.append(moments, z_moment)
        return (moments.transpose() @ np.eye(moments.size) @ moments) 
    
    # return ACF_GMM_val_grad(np.array([1.0,1.0]), ACF_data.to_numpy())
    solver = jaxopt.BFGS(fun = ACF_GMM_val_grad, verbose = False)
    res = solver.run(np.array([1.0,1.0]), data = ACF_data.to_numpy())
    beta_k, beta_l = res.params.tolist()
    numerical = optimize.minimize(lambda x: ACF_GMM_val_grad(x, ACF_data.to_numpy()), x0 = [1.0, 1.0], method="Nelder-Mead")
    beta_k_num, beta_l_num = numerical.x
    return(beta_k, beta_l, beta_k_num, beta_l_num)