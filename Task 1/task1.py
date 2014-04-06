import numpy as np
from numpy import matrix, sqrt, mean, concatenate, ndarray, dot, sum, transpose, std

import scipy as sp
from scipy.optimize import fmin_bfgs

from random import shuffle, randint
from datetime import datetime
from copy import deepcopy

import csv

fp = r'C:\Users\Jonny\Documents\York\CS\Year 3\MLAP\Open assessment\stock_price.csv'
fp_out = r'C:\Users\Jonny\Documents\York\CS\Year 3\MLAP\Open assessment\stock_price_mod.csv'

class FeatureExpander:
	def __init__(self, data_CV1, data_CV2, basic_feature_count):
		self.data_CV1 = data_CV1
		self.data_CV2 = data_CV2
		self.basic_feature_count = basic_feature_count
		
	def get_random_inclusion_list(self):
		inclusion_list = []
		
		for i in range(self.basic_feature_count):
			inclusion_list.append(randint(0,2))
			
		return inclusion_list
		
	def append_feature_expansions(self, data, inclusion_list):
		for i in range(len(data)):
			expanded = []
			
			expanded.append(1)
			
			for j in range(len(inclusion_list)):
				for k in range(inclusion_list[j]):
					expanded.append(data[i][2][j] ** k)
			
			data[i].append(matrix(expanded))
		
		return data
	
	def expand_features(self, inclusion_list):			
		expanded_CV1 = self.append_feature_expansions(deepcopy(self.data_CV1), inclusion_list)
		expanded_CV2 = self.append_feature_expansions(deepcopy(self.data_CV2), inclusion_list)
		
		return [expanded_CV1, expanded_CV2]

def append_features(source_data):
	for i in range(10, len(source_data)):
		# add feature matrix in next column
		row = source_data[i]
		feature_list = []
		
		prev_10_sv = [row[0] for row in source_data[i-10:i]]
		prev_10_sp = [row[1] for row in source_data[i-10:i]]
		
		# last change in sv
		feature_list.append(prev_10_sv[9] - prev_10_sv[8])
		
		# mean of prev 10 rows sv
		feature_list.append(mean(prev_10_sv))
		
		# std dev of prev 10 rows sv
		feature_list.append(std(prev_10_sv))
		
		# last sv
		feature_list.append(prev_10_sv[9])
		
		# last change in sp
		feature_list.append(prev_10_sp[9] - prev_10_sp[8])
		
		# mean of prev 10 rows sp
		feature_list.append(mean(prev_10_sp))
		
		# std dev of prev 10 rows sp
		feature_list.append(std(prev_10_sp))
		
		# last sp
		feature_list.append(prev_10_sp[9])
		
		source_data[i].append(feature_list)
	
	return source_data
	
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
	
	initial_THETA = matrix(np.ones([length_of_expansion,1]))
	
	return fmin_bfgs(total_squared_loss, initial_THETA, fprime=gradient, args=[data])

def split_data(all_data):
	shuffle(all_data)
	split = int(len(all_data)/2)
	return [all_data[:split], all_data[split:]]
	
def evaluate_MSE(data_CV1, data_CV2, THETA_CV1, THETA_CV2):
	squared_losses_CV1 = generate_squared_losses(THETA_CV2, data_CV1)
	squared_losses_CV2 = generate_squared_losses(THETA_CV1, data_CV2)
	
	total_squared_loss = sum(squared_losses_CV1) + sum(squared_losses_CV2)
	data_quantity = len(squared_losses_CV1) + len(squared_losses_CV2)
	
	return total_squared_loss / data_quantity
	
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
		
	all_sv = matrix([row[0] for row in data])
	
	return data
		
def linear(InputFileName):	
	raw_data = [[float(item) for item in row] for row in csv.reader(open(InputFileName, "rb"))]
	
	all_normalised_data = normalise(raw_data)
	
	training_data = append_features(raw_data)[10:]
	
	[data_CV1, data_CV2] = split_data(training_data)
	
	expander = FeatureExpander(data_CV1, data_CV2, 8)
	
	inclusion_list = []
	inclusion_list.append(4) # last change in sv
	inclusion_list.append(4) # mean of prev 10 rows sv
	inclusion_list.append(4) # std dev of prev 10 rows sv
	inclusion_list.append(4) # last sv
	inclusion_list.append(4) # last change in sp
	inclusion_list.append(4) # mean of prev 10 rows sp
	inclusion_list.append(4) # std dev of prev 10 rows sp
	inclusion_list.append(4) # last sp
	
	print inclusion_list
	
	[expanded_CV1, expanded_CV2] = expander.expand_features(inclusion_list)
	
	THETA_CV1 = regression(expanded_CV1)
	THETA_CV2 = regression(expanded_CV2)
	
	print evaluate_MSE(expanded_CV1,expanded_CV2,THETA_CV1,THETA_CV2)
	print THETA_CV1
	print THETA_CV2

print datetime.now()
linear(fp)
print datetime.now()