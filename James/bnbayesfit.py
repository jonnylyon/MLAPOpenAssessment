from __future__ import division
from datetime import datetime
from copy import deepcopy
from math import pow
import os
import csv

struct_fp = os.path.join(os.path.dirname(__file__), 'bnstruct.bn')
data_fp = os.path.join(os.path.dirname(__file__), 'bndata.csv')

def beta_mean(alpha, beta):
	return 1 / (1 + (beta / alpha))

def read_csv(file):
	return [[bool(int(c)) for c in row] for row in csv.reader(open(file, 'rb'))]

def get_all_bool_tuples(number_of_bools):
	result = [()]
	
	for i in range(number_of_bools):
		result_i = deepcopy(result)
		result = []
		for row in result_i:
			rt = row
			rt += True,
			rf = row
			rf += False,
			result.append(rt)
			result.append(rf)
	
	return result

def get_priors(struct):
	parents = [] # parents[i] is a tuple () of indexes of nodes linking directly to i
	for node in range(len(struct)):
		node_parents = ()
		for n in range(len(struct)):
			if struct[n][node] == True:
				node_parents += n,
		parents.append(node_parents)
	
	priors = [] # priors[i] is a dictionary {} mapping tuples of booleans to priors (beta tuples)
	for i, row in enumerate(struct):
		dict = {}
		for tuple in get_all_bool_tuples(len(parents[i])):
			dict[tuple] = (1,1)
		priors.append(dict)
	
	return (parents, priors)

def get_satisfying_data_rows(data, indexes, values):
	satisfying = []
	
	for row in data:
		include = True
		for i, index in enumerate(indexes):
			if row[index] != values[i]:
				include = False
		if include:
			satisfying.append(row)
	
	return satisfying

def calc_posterior(data, node, parents, params, prior):
	a0 = prior[0] # prior alpha
	b0 = prior[1] # prior beta
	
	occurences = {True: 0, False: 0}
	
	for row in get_satisfying_data_rows(data, parents, params):
		occurences[row[node]] += 1
	
	a1 = occurences[True]
	b1 = occurences[False]

	return (a0 + a1, b0 + b1)

def Beta(alpha, beta):
	return 1

def most_likely_theta(occurences):
	b = occurences[False]
	a = occurences[True]
	
	if b == 0: return 1
	
	return pow(a / (a + b), 1 / b)

def unnormalised_prob_theta_given_occurences(theta, occurences):
	return theta**occurences[True] * (1 - theta)**occurences[False]

def bnbayesfit(StructureFileName,DataFileName):
	struct = read_csv(StructureFileName)
	data = read_csv(DataFileName)
	
	[parents, priors] = get_priors(struct)
	
	posteriors = []
	
	for i, dict in enumerate(priors):
		post_dict = {}
		for params, prior in dict.iteritems():
			# print i, parents[i], params, prior
			post_dict[params] = calc_posterior(data, i, parents[i], params, prior)
		posteriors.append(post_dict)
	
	means = []
	
	for i, dict in enumerate(posteriors):
		mean_dict = {}
		for params, posterior in dict.iteritems():
			mean_dict[params] = beta_mean(posterior[0], posterior[1])
		means.append(mean_dict)
	
	return means

if __name__ == "__main__":
	print datetime.now()
	for row in bnbayesfit(struct_fp, data_fp):
		print row
	print datetime.now()