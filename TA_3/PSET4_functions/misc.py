import numpy as np
import pandas as pd
import jaxopt as jaxopt



def clean_data(data_set, num_prods):

    # Loading product characteristics
    
    X = pd.DataFrame(data_set['x1'], columns=['X1jm', 'X2jm', 'X3jm'])
    num_markets = int(X.shape[0]/num_prods)
    xi = pd.DataFrame(data_set['xi_all'], columns=['xijm'])
    X['m'] = X.groupby(X.index // num_prods).ngroup() 
    X['j'] = X.groupby('m').cumcount() 

    # Loading price and shares

    scols = [f"s{i}m" for i in range(1, num_prods+1)]
    pcols = [f"p{i}m" for i in range(1, num_prods+1)]
    S_long = pd.DataFrame(data_set['shares'].T.reshape(num_prods * num_markets,1), columns = ["sjm"])
    P_long = pd.DataFrame(data_set['P_opt'].T.reshape(num_prods * num_markets,1), columns = ["pjm"])

    # Loading marginal cost data
    
    w = pd.DataFrame(data_set['w'], columns=['wj'])
    z = pd.DataFrame(data_set['Z'], columns=['zjm'])
    eta = pd.DataFrame(data_set['eta'], columns=['etajm'])

    # Put it all together 

    cleaned_data = pd.concat([S_long, P_long, w, z, eta, xi, X], axis=1)
    return cleaned_data

def consumer_surplus_m(params, demand_features_m, v): 
    beta_1, beta_2, beta_3, alpha, sigma_alpha = params 
    alpha_i = alpha - sigma_alpha * v 
    u_ijm = demand_features_m @ np.array([beta_1, beta_2, beta_3, alpha_i, 1]).T
    return max(0, u_ijm.max())

def consumer_surplus(params, demand_features_long, random_vs, num_prods):
    num_markets = int(demand_features_long.shape[0]/num_prods) 
    total_cs = np.zeros((num_markets, 1))
    for m in range(num_markets):
        demand_features_m = demand_features_long[m : m + num_prods]
        total_cs_m = 0 
        for i in range(len(random_vs)):
            total_cs_m += consumer_surplus_m(params, demand_features_m, random_vs[i])
        total_cs[m] = total_cs_m
    
    return total_cs

def profits_m(gammas, supply_features_m, shares_m, prices_m, num_vs):
    gamma_0, gamma_1, gamma_2 = gammas 
    coef_vector = np.ascontiguousarray(np.array([gamma_0, gamma_1, gamma_2, 1]))
    feature_vector = np.concatenate((np.ones(supply_features_m.shape[0]).reshape(-1,1), supply_features_m), axis = 1)
    marginal_cost = feature_vector @ coef_vector.T 
    return num_vs * (shares_m * (prices_m - marginal_cost))

def marginal_cost_m(gammas, supply_features_m):
    gamma_0, gamma_1, gamma_2 = gammas 
    coef_vector = np.ascontiguousarray(np.array([gamma_0, gamma_1, gamma_2, 1]))
    feature_vector = np.concatenate((np.ones(supply_features_m.shape[0]).reshape(-1,1), supply_features_m), axis = 1)
    marginal_cost = feature_vector @ coef_vector.T 
    return marginal_cost

def profits(gammas, supply_features_long, shares_long, prices_long, num_prods, num_vs):
    num_markets = int(supply_features_long.shape[0]/num_prods)
    total_profits = np.zeros((num_markets, num_prods))
    for m in range(num_markets):
        supply_features_m = supply_features_long[m : m + num_prods]
        shares_m = shares_long[m : m + num_prods].flatten()
        prices_m = prices_long[m : m + num_prods].flatten()
        total_profits[m] = profits_m(gammas, supply_features_m, shares_m, prices_m, num_vs)
    return total_profits

def marginal_cost(gammas, supply_features_long, num_prods):
    num_markets = int(supply_features_long.shape[0]/num_prods)
    total_marginal_cost = np.zeros((num_markets, num_prods))
    for m in range(num_markets):
        supply_features_m = supply_features_long[m : m + num_prods]
        total_marginal_cost[m] = marginal_cost_m(gammas, supply_features_m)
    return total_marginal_cost