import numpy as np
import statsmodels.api as sm
import jax.numpy as jnp
import jaxopt as jaxopt
from PSET3_functions.misc import poly_2v, jnp_reg_predict, jnp_reg


def LP_estimation(dat):
    
    LP_data = dat.copy()
    #####
    # LP First Stage (from Stata Journal)
    #####

    LP_data, phi_vars = poly_2v("k", "m", LP_data)
    constant = jnp.ones((LP_data.shape[0], 1))
    phi_var_mat = jnp.array(LP_data[phi_vars].to_numpy())
    l = LP_data["l"]
    X = jnp.hstack((constant, phi_var_mat, l))
    y = LP_data["y"].copy()
    # lp_first_stage = sm.OLS(LP_data["y_gross"], sm.add_constant(LP_data[phi_vars + ["l"]])).fit()
    # lp_first_stage.params["l"]
    LP_data["y_gross_hat"] = jnp_reg_predict(y, X)
    beta_hat_stage1 = jnp_reg(y, X)
    beta_hat_l = beta_hat_stage1[len(beta_hat_stage1)]
    LP_data["phi_prediction"] = LP_data["y_gross_hat"] - beta_hat_l * LP_data["l"]

    #####
    # LP Second Stage 
    #####

        # making lags 
    LP_data["phi_prediction_lag"] = LP_data.groupby("firm_id")["phi_prediction"].shift(1)
    LP_data["k_lag"] = LP_data.groupby("firm_id")["k"].shift(1)
    LP_data["l_lag"] = LP_data.groupby("firm_id")["l"].shift(1)
    LP_data["m_lag"] = LP_data.groupby("firm_id")["m"].shift(1)
    LP_data["m_lag_2"] = LP_data.groupby("firm_id")["m_lag"].shift(1)

        # dropping NAs based on lags
    LP_data = LP_data[LP_data["m_lag"].notna()]
    LP_data = LP_data[LP_data["m_lag_2"].notna()]

        # index keeping, as we move to all numpy here so I can use the autograd
    LP_data_colnames = list(LP_data.columns)
    column_indices = {}
    for index, column_name in enumerate(LP_data_colnames):
        column_indices[column_name] = index

        # GMM objective function (that's auto-differentiable, hence the _grad)
    def LP_GMM_val_grad(params, data):
        beta_k, beta_m = params
        dat = data 
        omega = dat[:, column_indices["phi_prediction"]] - beta_k * dat[:, column_indices["k"]] - beta_m * dat[:, column_indices["m"]]
        omega_lag = dat[:, column_indices["phi_prediction_lag"]] - beta_k * dat[:, column_indices["k_lag"]] - beta_m * dat[:, column_indices["m_lag"]]
        g = omega_lag.reshape(-1, 1)
        g2 = jnp.power(omega_lag, 2).reshape(-1, 1)
        g3 = jnp.power(omega_lag, 3).reshape(-1, 1)
        g_func = jnp.hstack((jnp.ones((g.shape[0],1)),g, g2, g3))
        omega_hat = jnp_reg_predict(omega, g_func)
        ksi_epsilon = dat[:, column_indices["y_gross"]] - lp_first_stage.params["l"] * dat[:, column_indices["l"]] - beta_k * dat[:, column_indices["k"]] - beta_m * dat[:, column_indices["m"]] - omega_hat
        instruments = ["k", "m_lag", "l_lag", "m_lag_2", "k_lag"]
        moments = jnp.array([])
        for z in instruments:
            z_moment_col = dat[: , column_indices[z]] * ksi_epsilon
            z_moment = z_moment_col.mean()
            moments = jnp.append(moments, z_moment)
        return (moments.transpose() @ np.eye(moments.size) @ moments) 

    solver = jaxopt.BFGS(fun = LP_GMM_val_grad)
    res = solver.run(np.array([1.0,1.0]), data = LP_data.to_numpy())
    beta_k, beta_m = res.params.tolist()
    return(beta_k, beta_m, lp_first_stage.params["l"])