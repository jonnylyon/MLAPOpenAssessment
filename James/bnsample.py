from __future__ import division
from datetime import datetime
from bnbayesfit import bnbayesfit
from random import random
import os

struct_fp = os.path.join(os.path.dirname(__file__), 'bnstruct.bn')
data_fp = os.path.join(os.path.dirname(__file__), 'bndata.csv')

def instantiate_value(node, fittedbn, instantiation_dict):
    if instantiation_dict[node] == None:
        parent_vals = ()
        
        [parents, probabilities] = fittedbn[node]
        for parent in parents:
            if instantiation_dict[parent] == None:
                instantiation_dict = instantiate_value(parent, fittedbn, instantiation_dict)
            parent_vals += (instantiation_dict[parent],)
        
        probability = probabilities[parent_vals]
        
        rand = random()
        
        instantiation_dict[node] = (rand < probability)
        
    return instantiation_dict

def get_instantiation(fittedbn):
    instantiation_dict = {}
    
    for i in range(len(fittedbn)):
        instantiation_dict[i] = None
    
    for i in range(len(fittedbn)):
        instantiate_value(i, fittedbn, instantiation_dict)
    
    instantiation_array = []
    
    for i in range(len(fittedbn)):
        instantiation_array.append(int(instantiation_dict[i]))
    
    return instantiation_array

def bnsample(fittedbn,nsamples):
    instantiations = []
    for i in range(nsamples):
        instantiations.append(get_instantiation(fittedbn))
    return instantiations
    
if __name__ == "__main__":
    print datetime.now()
    fittedbn = bnbayesfit(struct_fp, data_fp)
    print bnsample(fittedbn, 10)
    
    print datetime.now()