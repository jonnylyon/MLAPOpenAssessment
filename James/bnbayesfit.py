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
    priors = [] # priors[i] is a list [] containing a parents tuple () and a priors dict {} (see below)
    
    # loop over all nodes to determine their parents and prior distributions
    for i, row in enumerate(struct):
        
        # work out which nodes are parents to this node and put them in a tuple
        parents = ()
        for n in range(len(struct)):
            if struct[n][i] == True:
                parents += n,
        
        # work out the prior distributions for each parameter and store them in a dict indexed
        # by a tuple containing the boolean values of each of the parents
        dict = {}
        for tuple in get_all_bool_tuples(len(parents)):
            dict[tuple] = (1,1)
        
        priors.append([parents,dict])
    
    return priors

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

def bnbayesfit(StructureFileName,DataFileName):
    struct = read_csv(StructureFileName)
    data = read_csv(DataFileName)
    
    priors = get_priors(struct)
    
    posteriors = []
    
    for i, [parents, dict] in enumerate(priors):
        posterior_dict = {}
        for params, prior in dict.iteritems():
            posterior_dict[params] = calc_posterior(data, i, parents, params, prior)
        posteriors.append([parents, posterior_dict])
    
    means = []
    
    for i, [parents, dict] in enumerate(posteriors):
        mean_dict = {}
        for params, posterior in dict.iteritems():
            mean_dict[params] = beta_mean(posterior[0], posterior[1])
        means.append([parents, mean_dict])
    
    return means

if __name__ == "__main__":
    print datetime.now()
    for row in bnbayesfit(struct_fp, data_fp):
        print row
    print datetime.now()