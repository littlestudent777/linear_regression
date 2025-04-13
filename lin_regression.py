import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize


# Least Absolute Deviations (LAD) function
def lav_loss(params, x, y):
    a, b = params
    return np.sum(np.abs(y - (a + b * x)))

# OLS
def build_ols(x, y):
    # OLS (Ordinary Least Squares)
    # The reshape method converts x into a matrix since LinearRegression expects a 2D array
    # The method trains a model on data by minimizing the sum of squared errors
    model_ols = LinearRegression().fit(x.reshape(-1, 1), y)
    a_ols, b_ols = model_ols.intercept_, model_ols.coef_[0]
    return a_ols, b_ols, model_ols

# LAD
def build_lav(x, y):
    initial_guess = [0, 0]  # Initial values for a and b
    result = minimize(lav_loss, initial_guess, args=(x, y))
    a_lav, b_lav = result.x
    return a_lav, b_lav
