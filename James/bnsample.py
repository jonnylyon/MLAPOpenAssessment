from __future__ import division
from datetime import datetime
from bnbayesfit import bnbayesfit
from random import random
import os

struct_fp = os.path.join(os.path.dirname(__file__), 'bnstruct.bn')
data_fp = os.path.join(os.path.dirname(__file__), 'bndata.csv')

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

# This is provided in case you wish to run the script directly
if __name__ == "__main__":
    print datetime.now()
    fittedbn = bnbayesfit(struct_fp, data_fp)
    print bnsample(fittedbn, 10)
    
    print datetime.now()