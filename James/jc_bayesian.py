from __future__ import division
from datetime import datetime
from copy import deepcopy
from math import pow
from random import random
import os
import csv

def beta_mean(alpha, beta):
    """ This method calculates the mean of the beta distribution
        with the alpha and beta parameters given
    """
    return alpha / (alpha + beta)

def read_csv(file):
    """ Reads the CSV file given and a list of lists representing
        the rows in the file.  Data is kept as booleans rather
        than 1s and 0s, for later convenience.
    """
    return [[bool(int(c)) for c in row] for row in csv.reader(open(file, 'rb'))]

def get_all_bool_tuples(number_of_bools):
    """ Given an integer n, this method constructs a list of tuples
        containing all 2^n permutations of true and false.  For example,
        if n (number_of_bools) = 2 then, the output is something like
        [(false, false), (false, true), (true, false), (true, true)]
        but not necessarily in that order.
    """
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
    """ This method constructs a data structure containing the parents nodes
        of each node, and a set of priors for the different combinations of
        values.  This data structure is formed as an outer list.  The nth
        element of the outer list refers to the nth node in the graph.
        The nth element of the outer list is another list containing at
        index 0 a tuple of n's parents, and at index 1 a dictionary mapping
        parent truth values to the prior beta distributions.
        
        All prior distributions have the values alpha = beta = 1.
        
        If node 0 has parent nodes 1 and 2, then the first element of the
        outer list (index 0) is the following list:
        [(1,2),
         {
          (false,false):(1,1),
          (false,true):(1,1),
          (true,false):(1,1),
          (true,true):(1,1)
         }
        ]
    """
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
    """ From the list of lists (rows) given as data,
        return only those rows which satisfy the
        criteria given.
        
        indexes is a list of node indexes.
        values is a list of the same length, containing
        the value (True/False) that the row must have for the
        node specified in the corresponding element of indexes
        
        i.e. a row 'satisfies' if:
            for all i in count(indexes), row[indexes[i]] = values[i]
    """
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
    """ Calculates and returns the (alpha, beta) of the
        posterior beta distribution for node given
        the parents and their corresponding values in params
        and the (alpha, beta) prior
    """
    a0 = prior[0] # prior alpha
    b0 = prior[1] # prior beta
    
    # dictionary of how many Trues and Falses occur
    occurences = {True: 0, False: 0}
    
    # loop over the rows that satisfy the parents/params conditions
    # specified
    for row in get_satisfying_data_rows(data, parents, params):
        # if the row has the value True for the given node, increment occurences[True],
        # otherwise increment occurences[False]
        occurences[row[node]] += 1
    
    a1 = occurences[True]
    b1 = occurences[False]
    
    # Return the (alpha, beta) of the posterior beta distribution
    return (a0 + a1, b0 + b1)

def bnbayesfit(StructureFileName,DataFileName):
    """ Reads the given Structure and Data files, and calculates
        the relevant posterior beta distributions for each node
        in the graph.
        
        Returns a data structure containing the means of the posterior
        beta distributions for each probability in the graph
    """
    struct = read_csv(StructureFileName)
    data = read_csv(DataFileName)
    
    # Get the prior beta distributions of all probabilities in the network
    # (Data structure described in comment for the get_priors method)
    priors = get_priors(struct)
    
    # Calculate the posterior beta distributions for each probability in
    # the network.
    posteriors = []
    for i, [parents, dict] in enumerate(priors):
        posterior_dict = {}
        for params, prior in dict.iteritems():
            posterior_dict[params] = calc_posterior(data, i, parents, params, prior)
        posteriors.append([parents, posterior_dict])
    
    # Calculate the means of each of the posterior beta distributions
    # in the network
    means = []
    for i, [parents, dict] in enumerate(posteriors):
        mean_dict = {}
        for params, posterior in dict.iteritems():
            mean_dict[params] = beta_mean(posterior[0], posterior[1])
        means.append([parents, mean_dict])
    
    # This section loops over the data and prints it in the format
    # of 'P(2=True|1=True,3=False) = 0.8', in case the information
    # is useful
    for i, [parents, mean_dict] in enumerate(means):
        for key, value in mean_dict.iteritems():
            parent_values = ["%s = %s" % (str(parents[j]), str(key[j])) for j in range(len(parents))]
            print "P(%s = True|%s) = %s" % (i, ', '.join(parent_values), value)
    
    # Returns a data structure similar to the one described in the
    # comment for the get_priors method, but where the dictionary
    # maps to the mean of the relevant posterior beta distribution,
    # rather than to the parameters of the prior beta distribution
    return means
    
####################################################################################################
### The following code was added for task 5
####################################################################################################
    
def instantiate_value(node, fittedbn, instantiation_dict):
    """ Instantiates a given node in the given network.  Note that instantiate_value
        is a recursive method.  If the value of a parent of the given node
        is not yet set, it recursively calls itself giving the unset parent
        as the node.
        If you call instantiate_value on a node that is already set then
        instantiate_value will not re-set it.
    """
    
    # Only continue if the given node has not yet been set,
    # otherwise just return the instantiation_dict unmodified
    if instantiation_dict[node] == None:
        # Get the values of the parents in this instantiation
        parent_vals = ()
        [parents, probabilities] = fittedbn[node]
        for parent in parents:
            # if the parent has not yet been instantiated,
            # instantiate it.  Otherwise use the existing
            # instantiation value.
            if instantiation_dict[parent] == None:
                instantiation_dict = instantiate_value(parent, fittedbn, instantiation_dict)
            parent_vals += (instantiation_dict[parent],)
        
        # get the probability given the values of the parents
        # in the instantiations
        probability = probabilities[parent_vals]
        
        # given the probability, calculate whether or not the
        # given node should be set to true or false, and set
        # the value in the dictionary
        rand = random()
        instantiation_dict[node] = (rand < probability)
        
    return instantiation_dict

def get_instantiation(fittedbn):
    """ Given a fittedbn output from bayesfit, this method
        returns a single instantiation of fittedbn.
        
        This instantiation is in the format of a list, where the nth element
        of the list represents the truth (1) or falsity(0)
        of node n.
    """
    
    # During the course of computing the instantiation, the
    # instantiation is held as a dictionary, mapping the integer
    # index of each node to True/False as appropriate (all
    # initialised to None)
    instantiation_dict = {}
    
    for i in range(len(fittedbn)):
        instantiation_dict[i] = None
    
    # Instantiate each node in the network.  Note that instantiate_value
    # is a recursive method.  If the value of a parent of the given node
    # is not yet set, it recursively calls itself giving the unset parent
    # as the node.  Therefore it does not matter in which order we iterate
    # over the nodes at this stage.  instantiate_value just handles it.
    # If we call instantiate_value on a node that is already set (i.e. it
    # is the parent of one that we have already tried to set) then
    # instantiate_value will not re-set it.
    for i in range(len(fittedbn)):
        instantiate_value(i, fittedbn, instantiation_dict)
    
    # Once the instantiation dictionary has all been filled out, we convert
    # it to a list to return, as required.    
    instantiation_list = []
    for i in range(len(fittedbn)):
        instantiation_list.append(int(instantiation_dict[i]))
    
    return instantiation_list

def bnsample(fittedbn,nsamples):
    """ This is method, given a fittedbn output from bayesfit
        and an integer nsamples, returns a list of instantiations.
        An instantiation is itself a list, where the nth element
        of the list represents the truth (1) or falsity(0)
        of node n.
    """
    instantiations = []
    for i in range(nsamples):
        instantiations.append(get_instantiation(fittedbn))
    return instantiations