import numpy as np
import statsmodels.api as sm
from scipy import optimize
import jaxopt as jaxopt
from PSET3_functions.misc import poly_2v 


def OP_estimation(dat):

    OP_data = dat.copy()
    # making lags (that we'll use later)
    OP_data["k_lag"] = OP_data.groupby("firm_id")["k"].shift(1)
    OP_data["i_lag"] = OP_data.groupby("firm_id")["i"].shift(1)
    
    #####
    # OP First Stage
    #####

    OP_data, phi_vars = poly_2v("k", "i", OP_data)
    op_first_stage = sm.OLS(OP_data["y"], sm.add_constant(OP_data[phi_vars + ["l"]])).fit()

    OP_data["y_hat"] = op_first_stage.predict(sm.add_constant(OP_data[phi_vars + ["l"]]))
    OP_data["phi_prediction"] = OP_data["y_hat"] - op_first_stage.params["l"] * OP_data["l"]
    OP_data["phi_prediction_lag"] = OP_data.groupby("firm_id")["phi_prediction"].shift(1)

    #####
    # OP Second Stage 
    #####

        # making survival dummy
    OP_data['survival_dummy'] = OP_data.groupby('firm_id')['firm_id'].transform(lambda x: x.shift(-1)).notnull().astype(int)

        # predicted exit (logistic regression on capital and investment)
    survival_predict = sm.Logit(OP_data["survival_dummy"], sm.add_constant(OP_data[phi_vars])).fit(disp=0)
    OP_data["p_s_hat"] = survival_predict.predict(sm.add_constant(OP_data[phi_vars]))
    OP_data["p_s_hat"].head()
    OP_data.describe()

        # dropping NAs based on lags
    OP_data = OP_data[OP_data["phi_prediction_lag"].notna()]
    OP_data = OP_data[OP_data["k_lag"].notna()]

        # Second stage objective function
    def OP_GMM_val(beta_k, data):
        dat = data 
        dat["omega_t"] = dat["phi_prediction"] - beta_k * dat["k"] 
        dat["omega_lag"] = dat["phi_prediction_lag"] - beta_k * dat["k_lag"] 
        dat, g_vars = poly_2v("omega_lag","p_s_hat", dat)
        omega_hat_reg = sm.OLS(dat["omega_t"], sm.add_constant(dat[g_vars])).fit()
        dat["omega_hat"] = omega_hat_reg.predict(sm.add_constant(dat[g_vars]))
        dat["ksi_epsilon"] = dat["y"] - op_first_stage.params["l"] * dat["l"] - beta_k * dat["k"] - dat["omega_hat"]
        dat["ksi_epsilon_square"] = np.square(dat["y"] - op_first_stage.params["l"] * dat["l"] - beta_k * dat["k"] - dat["omega_hat"]) 
        return dat["ksi_epsilon_square"].mean() #minimizing E[squared errors]; basically OLS, most stable moments condition I found and from Petrin Poi Stata Journal Article
    
    beta_k = optimize.minimize(lambda x: OP_GMM_val(x, OP_data), x0 = 1, method="Nelder-Mead") 
    return (beta_k.x[0], op_first_stage.params["l"])