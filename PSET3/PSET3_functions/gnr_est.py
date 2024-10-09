import numpy as np
from scipy import optimize
import jax.numpy as jnp
from jax import jacfwd
import jaxopt as jaxopt
import sympy as sp
from PSET3_functions.misc import poly_2v, poly_3v, jnp_reg_predict

def GNR_estimation(dat):

    GNR_data = dat.copy() 
    
    GNR_data["s"] = np.log( (np.exp(GNR_data["m"]) * np.exp(GNR_data["p_m"])) / (np.exp(GNR_data["y"]) * np.exp(GNR_data["p_y"])))
    GNR_data['l_lag'] = GNR_data.groupby('firm_id')['l'].shift(1)
    GNR_data['k_lag'] = GNR_data.groupby('firm_id')['k'].shift(1)

    #####
    # GNR First Stage 
    #####

    GNR_data, gamma_vars = poly_3v("l", "k", "m", GNR_data)
    GNR_data['1'] = 1
    gamma_vars = ["1"] + gamma_vars
    GNR_data.describe()
    len(gamma_vars)

        # index keeping, as we move to all numpy to speed things up
    GNR_data_colnames = list(GNR_data.columns)
    column_indices = {}
    for index, column_name in enumerate(GNR_data_colnames):
        column_indices[column_name] = index

    def GNR_stage1(gamma):
        dat = GNR_data.to_numpy()
        est_term = dat[:,[column_indices[column_name] for column_name in gamma_vars]]@gamma
        est_term = jnp.array(est_term)
        est_term_log = jnp.log(est_term)
        val = dat[:, column_indices["s"]] -  est_term_log
        return val

    GNR_stage1_grad = jacfwd(GNR_stage1)

    gamma_0 = np.ones(len(gamma_vars))*.01

    prediction = optimize.least_squares(GNR_stage1, gamma_0, jac = GNR_stage1_grad)
    GNR_data['stage1_residuals'] = prediction.fun
    np.exp(prediction.fun)
    E_Epsilon = np.exp(prediction.fun).mean()
    gamma_hats = prediction.x / E_Epsilon

    gamma_ests = dict(zip(gamma_vars, gamma_hats))
    GNR_data['input_elasticity'] = GNR_data[list(gamma_ests.keys())]@list(gamma_ests.values())

    #####
    # GNR Second Stage 
    #####

    # Initialize the integrated dictionary
    integrated_gamma_ests = dict()

    # Integrate each term in the coefficients_dict and update the coefficients_dict
    for term, coeff in gamma_ests.items():
        
        # Parse the term using sympy
        term = term.replace('^', '**')
        term = term.replace (' ', '*')
        parsed_term = sp.sympify(term.replace('^', '**'))

        # Integrate the parsed term with respect to x
        integrated_term = sp.integrate(parsed_term, sp.symbols('m'))
        new_term = coeff * integrated_term
        new_terms = sp.Mul.make_args(new_term)
        new_coef, rest = (new_terms[0], sp.Mul(*new_terms[1:]))
        
        #solve for coefficient by setting all terms = 1
        substitution_dict = {symbol: 1 for symbol in new_term.free_symbols}
        new_coeff = new_term.subs(substitution_dict) 
        
        # Add the integrated term multiplied by its coefficient to the integrated_expression
        integrated_gamma_ests[str(rest)] = new_coeff

    expressions=list(integrated_gamma_ests.keys())
    k, l, m = sp.symbols('k l m')
    expression_functions = [sp.lambdify((k, l, m), expr) for expr in expressions]
    for idx, func in enumerate(expression_functions):
        column_name = str(expressions[idx])  # Convert the expression to string for column name
        GNR_data[column_name] = func(GNR_data['k'], GNR_data['l'], GNR_data['m']) #generate corresponding column

    GNR_data['integrated_elasticity'] = np.float32(GNR_data[integrated_gamma_ests.keys()]@list(integrated_gamma_ests.values())) #create integrated elasticity
    GNR_data['big_Y'] = GNR_data['y'] - GNR_data['integrated_elasticity'] - GNR_data['stage1_residuals'] #big Y definition from paper
    GNR_data['big_Y_lag'] = GNR_data.groupby('firm_id')['big_Y'].shift(1) #lag it for the GMM value function

    # dropping NAs based on lags
    GNR_data = GNR_data[GNR_data["big_Y_lag"].notna()]
    GNR_data = GNR_data[GNR_data["l_lag"].notna()]

    # making polynomial to predict integration constant
    GNR_data, alpha_vars = poly_2v("l", "k", GNR_data)
    GNR_data, alpha_vars_lag = poly_2v('l_lag', 'k_lag', GNR_data)

    # dropping extraneous columns
    GNR_data = GNR_data.drop(columns=GNR_data.filter(like='copy'))
    GNR_data = GNR_data.drop(columns=GNR_data.filter(like='*'))

    # index keeping, as we move to all numpy to autodiff
    GNR_data_colnames = list(GNR_data.columns)
    column_indices = {}
    for index, column_name in enumerate(GNR_data_colnames):
        column_indices[column_name] = index

    # GNR GMM function, in form Jax can auto-differentiate
    def GNR_GMM_val_grad(alpha, data):
        dat = data 
        omega = dat[:, column_indices['big_Y']] + dat[:,[column_indices[column_name] for column_name in alpha_vars]]@alpha
        omega_lag = dat[:, column_indices["big_Y_lag"]] + dat[:,[column_indices[column_name] for column_name in alpha_vars_lag]]@alpha
        g = omega_lag
        g2 = jnp.power(omega_lag, 2)
        g3 = jnp.power(omega_lag, 3)
        g_func = jnp.array([jnp.ones(g.shape[0]),g, g2, g3])
        omega_hat = jnp_reg_predict(omega, g_func)
        ksi = omega - omega_hat
        moments = jnp.array([])
        for z in alpha_vars:
            z_moment_col = dat[: , column_indices[z]] * ksi
            z_moment = z_moment_col.mean()
            moments = jnp.append(moments, z_moment)
        return (moments.transpose() @ np.eye(moments.size) @ moments) 

    alpha_0 = jnp.ones(len(alpha_vars))*.01
    solver = jaxopt.BFGS(fun = GNR_GMM_val_grad)

    res = solver.run(alpha_0, data = GNR_data.to_numpy())
    real_omegas = GNR_data['big_Y'] + GNR_data[alpha_vars]@res.params
    return (GNR_data['input_elasticity'].mean(), real_omegas.mean())