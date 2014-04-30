import numpy as np
from numpy import matrix, mean, sum, std, nditer, transpose

from random import shuffle, randint
from copy import deepcopy

import csv

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
                    expanded.append(data[i][-1][j] ** k)
            
            data[i].append(matrix(expanded))
        
        return data
    
    def expand_features(self, inclusion_list):
        expanded = self.append_feature_expansions(deepcopy(self.data), inclusion_list)
        
        return expanded

def append_features(source_data):
    result_data=[]
    
    for i in range(10, len(source_data)):
        # add feature matrix in next column
        this_row = source_data[i]
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

        this_row.append(feature_list)
        result_data.append(this_row)

    return result_data

def split_data(all_data):
    shuffle(all_data)
    split = int(len(all_data)/2)
    return [all_data[:split], all_data[split:]]
    
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

def sign(THETA):
    shape = THETA.shape
    out = [0 if theta == 0 else theta / abs(theta) for theta in nditer(THETA)]
    out = matrix(out)
    out = np.reshape(out, shape)
    return out

def write_to_file(data, filepath):
    lines = []
    for row in data:
        cells = []
        for item in row:
            cells.append(str(item))
    
        line = ",".join(cells)
        lines.append(line + "\n")
        
    file = open(filepath, 'w')
    file.writelines(lines)
    file.close()

def load_data(filepath):
    return [[float(item) for item in row] for row in csv.reader(open(filepath, "rb"))]