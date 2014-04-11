from __future__ import division

import numpy as np
from numpy import matrix, sqrt, mean, concatenate, ndarray, dot, sum, transpose, std, exp, reshape, nditer, multiply

import scipy as sp
from scipy.optimize import fmin_bfgs

from random import shuffle, randint
from datetime import datetime
from copy import deepcopy
from math import log
from sys import float_info

import csv

fp_out = r'C:\Users\Jonny\Documents\York\CS\Year 3\MLAP\Open assessment\MLAPOpenAssessment\stock_price_mod.csv'
fp = r'C:\Users\Jonny\Documents\York\CS\Year 3\MLAP\Open assessment\MLAPOpenAssessment\stock_price.csv'

class FeatureExpander:
	def __init__(self, data):
		self.data = data
		
	def get_random_inclusion_list(self, feature_count):
		inclusion_list = []
		
		for i in range(feature_count):
			inclusion_list.append(randint(0,2))
			
		return inclusion_list
		
	def append_feature_expansions(self, data, inclusion_list):
		for i in range(len(data)):
			expanded = []
			
			expanded.append(1)
			
			for j in range(len(inclusion_list)):
				for k in range(1, inclusion_list[j] + 1):
					expanded.append(data[i][3][j] ** k)
			
			data[i].append(matrix(expanded))
		
		return data
	
	def expand_features(self, inclusion_list):			
		expanded = self.append_feature_expansions(deepcopy(self.data), inclusion_list)
		
		return expanded

def append_classifications(source_data):
	result_data=[]
	
	for i in range(len(source_data)):
		this_row = source_data[i]
		last_row = source_data[i-1]
		this_row.append(classify(last_row[1], this_row[1]))
		result_data.append(this_row)
	
	return result_data

def append_features(source_data):
	result_data=[]
	
	for i in range(10, len(source_data)):
		# add feature matrix in next column
		this_row = source_data[i]
		feature_list = []
		
		prev_10_sv = [row[0] for row in source_data[i-10:i]]
		prev_10_sp = [row[1] for row in source_data[i-10:i]]
		
		# # last change in sv
		# feature_list.append(prev_10_sv[9] - prev_10_sv[8])
		
		# # mean of prev 10 rows sv
		# feature_list.append(mean(prev_10_sv))
		
		# # std dev of prev 10 rows sv
		# feature_list.append(std(prev_10_sv))
		
		# # last sv
		# feature_list.append(prev_10_sv[9])
		
		# # last change in sp
		# feature_list.append(prev_10_sp[9] - prev_10_sp[8])
		
		# # mean of prev 10 rows sp
		# feature_list.append(mean(prev_10_sp))
		
		# # std dev of prev 10 rows sp
		# feature_list.append(std(prev_10_sp))
		
		# # last sp
		# feature_list.append(prev_10_sp[9])
		
		feature_list.append(prev_10_sp[9])
		feature_list.append(prev_10_sp[9] - prev_10_sp[0])
		feature_list.append(prev_10_sp[9] - prev_10_sp[8])
		feature_list.append(prev_10_sp[9] - prev_10_sp[0])
		feature_list.append(std(prev_10_sp))
		feature_list.append(sum([prev_10_sp[j] * prev_10_sv[j] for j in range(10)]))
		feature_list.append(mean(prev_10_sp))
		feature_list.append(mean(prev_10_sv))

		this_row.append(feature_list)
		result_data.append(this_row)

	return result_data
	
def unflatten_theta(THETA):
	if len(THETA.shape) > 1:
		return THETA
		
	return matrix(reshape(THETA, (-1, 5)))
	
def log_sum_exp(XTHETA):
	max = XTHETA[0,0]
	for elem in nditer(XTHETA):
		if elem > max:
			max = elem
	
	sum = 0
	
	for elem in nditer(XTHETA):
		sum += exp(elem - max)

	return (max + log(sum))
	
def log_likelihood(THETA, data):
	result = 0.0
	THETA = unflatten_theta(THETA)
	for i in range(len(data)):	
		X = data[i][4]
		c = data[i][2]
		result += dot(X, THETA[:,c])[0,0]
		result -= log_sum_exp(dot(X, THETA))
	
	return result

def neg_log_likelihood(THETA, data):
	return log_likelihood(THETA, data) * -1

def prob_a_given_X(THETA, X, a):
	dot_Xi_Ta = dot(X, THETA[:,a])[0,0]
	l_s_e = log_sum_exp(dot(X, THETA))
	
	if l_s_e > 700:
		return float_info.min
	else:
		numerator = exp(dot_Xi_Ta)
		denominator = exp(l_s_e)
		return numerator / denominator

def classify(last_price, this_price):
	percent_change = this_price / last_price
	
	if percent_change < 0.9:
		return 4
	if percent_change < 0.95:
		return 2
	if percent_change <= 1.05:
		return 0
	if percent_change <= 1.1:
		return 1
	
	return 3
		
def gradient(THETA, data):
	THETA = unflatten_theta(THETA)
	
	GRAD_shape = THETA.shape
	if len(GRAD_shape) == 1:
		GRAD_shape = [1, THETA.shape[0]]
	GRAD = np.zeros(GRAD_shape)
	GRAD[:,:] = float_info.min
	
	for a in range(5):
		T_a = THETA[:,a]
		
		for i in range(len(data)):
			X_i = data[i][4]
			c = data[i][2]
			T_c = THETA[:,c]
			I = 1 if a == c else 0
			delta_loss = multiply(X_i, I - prob_a_given_X(THETA, X_i, a))
			delta_loss = [float(elem) for elem in nditer(delta_loss)]
			GRAD[:,a] += delta_loss
	
	flattened = np.reshape(sp.array(GRAD), (-1))
	
	for i, elem in enumerate(flattened):
		if elem == 0:
			flattened[i] = float_info.min
	
	return np.reshape(sp.array(GRAD), (-1))

def regression(data):
	length_of_expansion = data[0][4].shape[1]
	
	initial_THETA = matrix(np.ones([length_of_expansion,5]))
	
	return fmin_bfgs(neg_log_likelihood, initial_THETA, fprime=gradient, args=[data])

def split_data(all_data):
	shuffle(all_data)
	split = int(len(all_data)/2)
	return [all_data[:split], all_data[split:]]

def hard_predict(THETA, X):
	best_i = -1
	p_best = -1
	for i in range(5):
		p_i = prob_a_given_X(THETA, X, i)
		if prob_a_given_X(THETA, X, i) > p_best:
			p_best = p_i
			best_i = i
	
	return best_i

def percentage_correct_classifications(THETA, data):
	correct = 0
	incorrect = 0
	
	for row in data:
		actual_classification = row[2]
		hard_prediction = hard_predict(THETA, row[4])
		
		if hard_prediction == actual_classification:
			correct += 1
		else:
			incorrect += 1
	
	return (100 * correct) / (correct + incorrect)
	
def evaluate(data_CV1, data_CV2, THETA_CV1, THETA_CV2):
	pc_CV1 = percentage_correct_classifications(unflatten_theta(THETA_CV2), data_CV1)
	pc_CV2 = percentage_correct_classifications(unflatten_theta(THETA_CV1), data_CV2)
	
	return [(pc_CV1 + pc_CV2) / 2]
	
def normalise(data):
	all_sv = [row[0] for row in data]
	all_sp = [row[1] for row in data]
	
	sv_std = std(all_sv)
	sv_mean = mean(all_sv)
	
	sp_std = std(all_sp)
	sp_mean = mean(all_sp)
	
	for row in data:
		row[0] = (row[0] - sv_mean) / sv_std
		row[1] = (row[1] - sp_mean) / sp_std
	
	return data

def write_to_file(data):
	lines = []
	for row in data:
		cells = []
		for item in row:
			cells.append(str(item))
	
		line = ",".join(cells)
		lines.append(line + "\n")
		
	file = open(fp_out, 'w')
	file.writelines(lines)
	file.close()	
		
def logistic(InputFileName):	
	raw_data = [[float(item) for item in row] for row in csv.reader(open(InputFileName, "rb"))]
	
	#all_normalised_data = normalise(raw_data)
	all_normalised_data = raw_data
	all_normalised_data = append_classifications(all_normalised_data)
	training_data = append_features(all_normalised_data)
	expander = FeatureExpander(training_data)
	
	inclusion_list = []
	inclusion_list.append(1) # last change in sv
	inclusion_list.append(0) # mean of prev 10 rows sv
	inclusion_list.append(4) # std dev of prev 10 rows sv
	inclusion_list.append(0) # last sv
	inclusion_list.append(0) # last change in sp
	inclusion_list.append(0) # mean of prev 10 rows sp
	inclusion_list.append(0) # std dev of prev 10 rows sp
	inclusion_list.append(0) # last sp
	
	expanded = expander.expand_features(inclusion_list)
	
	write_to_file(expanded)
	
	[expanded_CV1, expanded_CV2] = split_data(expanded)
	
	THETA_CV1 = regression(expanded_CV1)
	THETA_CV2 = regression(expanded_CV2)
	
	print THETA_CV1
	print THETA_CV2
	return evaluate(expanded_CV1,expanded_CV2,THETA_CV1,THETA_CV2)

print datetime.now()
print logistic(fp)
print datetime.now()