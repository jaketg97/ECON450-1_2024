import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import jax.numpy as jnp
import jaxopt as jaxopt


####
# Defining some helper functions, to set me up for the work to come
####

# gen_data function, vars of interest and return a dataset of them
def gen_data(data):
    x = data.copy()

    #value added output
    x["y_gross"] = x["X03"]
    x["m"] = x["X44"]
    x["p_y"] = x["X45"]
    x["p_m"] = x["X49"]
    x["y"] = np.log(np.exp(x["y_gross"] + x["p_y"]) - np.exp(x["m"] + x["p_m"]))
    
    #other vars
    x["k"] = x["X40"]
    x["i"] = x["X39"]
    x["l"] = x["X43"]
    
    interest = ["year", "firm_id", "y_gross", "y", "k", "i", "l", "m", "p_m", "p_y"]
    return x[interest]

# shortcut functions to quickly estimate 3rd order polynomials of one/two variables in our dataframe (we'll do that a lot)

def poly_1v(x, data):
    poly = PolynomialFeatures(3)
    out = poly.fit_transform(data[x].to_numpy().reshape(-1,1))
    out = pd.DataFrame(out, columns=["constant", x, x + "_2", x + "_3"])
    for i in [x, x + "_2", x + "_3"]:
        data[i] = out[i].to_numpy()
    return [x, x + "_2", x + "_3"]

def poly_2v(x, y, data):
    poly = PolynomialFeatures(3)
    poly.n_features_in_ = 2  
    poly.feature_names_in_ = [x, y]
    poly.set_output(transform="pandas")
    out = poly.fit_transform(data[[x,y]])
    out = out.drop(columns = ["1", x, y])
    out = data.join(out, rsuffix = 'copy')
    names = list(poly.get_feature_names_out())
    names.remove('1')
    return out, names

def poly_3v(x, y, z, data):
    poly = PolynomialFeatures(3)
    poly.n_features_in_ = 3  
    poly.feature_names_in_ = [x, y, z]
    poly.set_output(transform="pandas")
    out = poly.fit_transform(data[[x,y, z]])
    out = out.drop(columns = ["1", x, y, z])
    out = data.join(out, rsuffix = '_copy')
    names = list(poly.get_feature_names_out())
    names.remove('1')
    return out, names

# shortcut functions to do everything in jnp so I can use autograd

def jnp_reg(y, X):
    return jnp.linalg.solve(X.dot(X.T), X.dot(y))

def jnp_reg_predict(y, X):
    return jnp.linalg.solve(X.dot(X.T), X.dot(y)).dot(X)

def boot(data, func, reps):
    coefs = list()
    for i in range(reps):
        temp = data.groupby("firm_id").sample(replace = True, random_state=i, frac = 1)
        result = func(temp)
        coefs.append(result)
    return coefs