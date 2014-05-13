from __future__ import division
import os

import numpy as np
import scipy as sp

from generics import FeatureExpander, append_features, split_data_3_folds, normalise, write_to_file, load_data, sign
from datetime import datetime
from numpy import matrix, dot, ndarray, nditer, transpose
from scipy.optimize import fmin_bfgs

fp = os.path.join(os.path.dirname(__file__), 'stock_price.csv')
fp_out = os.path.join(os.path.dirname(__file__), 'stock_price_mod.csv')
    
def generate_squared_losses(THETA, data):
    squared_losses = ndarray(shape=(len(data),1))
    
    for i in range(len(data)):
        squared_losses[i] = (data[i][1] - dot(data[i][3],THETA))**2
    
    return squared_losses
    
def total_squared_loss(THETA, data):
    return sum(generate_squared_losses(THETA, data))
    
def reg_loss_lasso(THETA, data, lamb, sign_func = None):
    complexity = 0
    
    for theta in nditer(THETA):
        complexity += abs(theta)
    
    return (lamb * total_squared_loss(THETA, data)) + ((1 - lamb) * complexity)

def gradient_lasso(THETA, data, lamb, sign_func):
    # This THETA_0 business is necessary because fmin_bfgs flattens THETA into
    # an array which is useless for us
    THETA_0 = []
    for elem in nditer(THETA):
        THETA_0.append(float(elem))
    THETA_0 = transpose(matrix(THETA_0))
    
    X_matrix = matrix([[float(elem) for elem in nditer(row_matrix)] for row_matrix in [row[-1] for row in data]])
    y_matrix = transpose(matrix([row[1] for row in data]))
    
    first_part = -2 * lamb * dot(transpose(X_matrix), y_matrix - dot(X_matrix, THETA_0))
    second_part = ((1 - lamb) * sign_func(THETA_0))
    
    grad_matrix = first_part + second_part
    
    return sp.array(grad_matrix.A1)

def regression(data, lamb):
    length_of_expansion = data[0][-1].shape[1]
    
    initial_THETA = matrix(np.ones([length_of_expansion,1]))
    
    return fmin_bfgs(reg_loss_lasso, initial_THETA, fprime=gradient_lasso, args=[data,lamb,sign])

def evaluate(data_CV1, data_CV2, THETA_CV1, THETA_CV2):
    mse_CV1 = calc_mse(THETA_CV2, data_CV1)
    mse_CV2 = calc_mse(THETA_CV1, data_CV2)
    
    return (mse_CV1 + mse_CV2) / 2
    
def calc_mse(data, THETA):
    squared_losses = generate_squared_losses(THETA, data)
    return sum(squared_losses) / len(squared_losses)
        
def reglinear(InputFileName, inclusion_list = None):
    raw_data = load_data(InputFileName)
    
    training_data = append_features(raw_data)
        
    expander = FeatureExpander(training_data)
    
    if not inclusion_list:
        inclusion_list = []
        inclusion_list.append(1) # last sv
        inclusion_list.append(1) # last change in sv
        inclusion_list.append(0) # mean of prev 10 rows sv
        inclusion_list.append(1) # std dev of prev 10 rows sv
        inclusion_list.append(0) # last sp
        inclusion_list.append(0) # last change in sp
        inclusion_list.append(2) # mean of prev 10 rows sp
        inclusion_list.append(0) # std dev of prev 10 rows sp
    
    expanded = expander.expand_features(inclusion_list)
    
    write_to_file(expanded, fp_out)
    
    [expanded_CV1, expanded_CV2, expanded_test] = split_data_3_folds(expanded)
    
    results = []
    
    lamb_resolution = 100
    for lamb in [i/lamb_resolution for i in range(1,lamb_resolution)]:
        print lamb
        THETA_CV1 = regression(expanded_CV1, lamb)
        THETA_CV2 = regression(expanded_CV2, lamb)
        
        results.append((THETA_CV1, lamb, calc_mse(expanded_CV2, THETA_CV1)))
        results.append((THETA_CV2, lamb, calc_mse(expanded_CV1, THETA_CV2)))
    
    best_result = results[0]
    for result in results:
        if result[2] < best_result[2]:
            best_result = result
    
    final_score = calc_mse(expanded_test, best_result[0])
    
    return (best_result, final_score)
    
if __name__ == "__main__":
    print datetime.now()
    print reglinear(fp)
    print datetime.now()