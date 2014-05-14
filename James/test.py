from jc_bayesian import bnbayesfit, bnsample
import os
from math import fabs

struct_fp = os.path.join(os.path.dirname(__file__), 'bnstruct.bn')
data_fp = os.path.join(os.path.dirname(__file__), 'bndata.csv')

fittedbn = bnbayesfit(struct_fp, data_fp)

for i in xrange(1,100):
    p = i*100
    sample = bnsample(fittedbn, p)
    cond1 = [s for s in sample if s[6] == 1]
    cond2 = [s for s in cond1 if s[7] == 1]
    
    est = len(cond2) / float(len(cond1))
    act = 0.414988185
    
    print p, 100 * abs(est - act)/act