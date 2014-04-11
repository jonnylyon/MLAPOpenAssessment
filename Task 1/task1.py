import csv

import numpy as np
import scipy as sp

from generics import FeatureExpander, append_features, split_data, normalise, write_to_file
from datetime import datetime
from numpy import matrix, dot, ndarray
from scipy.optimize import fmin_bfgs

fp_out = r'C:\Users\Jonny\Documents\York\CS\Year 3\MLAP\Open assessment\MLAPOpenAssessment\stock_price_mod.csv'
fp = r'C:\Users\Jonny\Documents\York\CS\Year 3\MLAP\Open assessment\MLAPOpenAssessment\stock_price.csv'
	
def generate_squared_losses(THETA, data):
	squared_losses = ndarray(shape=(len(data),1))
	
	for i in range(len(data)):
		squared_losses[i] = (data[i][1] - dot(data[i][3],THETA))**2
	
	return squared_losses
	
def total_squared_loss(THETA, data):
	return sum(generate_squared_losses(THETA, data))
	
def mean_squared_loss(THETA, data):
	return total_squared_loss(THETA, data) / len(data)

def gradient(THETA, data):
	GRAD = []
	
	for i in range(len(THETA)):
		GRAD.append(0.0)
	
	for i in range(len(data)):
		X_i = data[i][3]
		sp_computed = dot(X_i, THETA)
		sp_actual = data[i][1]
		
		error = (sp_computed - sp_actual)[0,0]

		for j in range(len(THETA)):
			GRAD[j] += error * X_i[0,j]
	
	return sp.array(GRAD)

def regression(data):
	length_of_expansion = data[0][3].shape[1]
	
	initial_THETA = matrix(np.zeros([length_of_expansion,1]))
	
	return fmin_bfgs(total_squared_loss, initial_THETA, fprime=gradient, args=[data])

def evaluate_MSE(data_CV1, data_CV2, THETA_CV1, THETA_CV2):
	squared_losses_CV1 = generate_squared_losses(THETA_CV2, data_CV1)
	squared_losses_CV2 = generate_squared_losses(THETA_CV1, data_CV2)
	
	total_squared_loss = sum(squared_losses_CV1) + sum(squared_losses_CV2)
	data_quantity = len(squared_losses_CV1) + len(squared_losses_CV2)
	
	return total_squared_loss / data_quantity
		
def linear(InputFileName):	
	raw_data = [[float(item) for item in row] for row in csv.reader(open(InputFileName, "rb"))]
	
	all_normalised_data = normalise(raw_data)
	#all_normalised_data = raw_data
	
	training_data = append_features(all_normalised_data)
		
	expander = FeatureExpander(training_data)
	
	inclusion_list = []
	inclusion_list.append(0) # last change in sv
	inclusion_list.append(0) # mean of prev 10 rows sv
	inclusion_list.append(0) # std dev of prev 10 rows sv
	inclusion_list.append(0) # last sv
	inclusion_list.append(0) # last change in sp
	inclusion_list.append(0) # mean of prev 10 rows sp
	inclusion_list.append(0) # std dev of prev 10 rows sp
	inclusion_list.append(0) # last sp
	
	expanded = expander.expand_features(inclusion_list)
	
	write_to_file(expanded, fp_out)
	
	[expanded_CV1, expanded_CV2] = split_data(expanded)
	
	THETA_CV1 = regression(expanded_CV1)
	THETA_CV2 = regression(expanded_CV2)
	
	result = evaluate_MSE(expanded_CV1,expanded_CV2,THETA_CV1,THETA_CV2)
	
	print result
	print THETA_CV1
	print THETA_CV2
	
	return result

print datetime.now()
linear(fp)
print datetime.now()