from __future__ import division
from bnsample import bnsample
from bnbayesfit import bnbayesfit
import os

struct_fp = os.path.join(os.path.dirname(__file__), 'bnstruct.bn')
data_fp = os.path.join(os.path.dirname(__file__), 'bndata.csv')

fittedbn = bnbayesfit(struct_fp, data_fp)
sample = bnsample(fittedbn, 5000)

satisfies_precondition = []

for inst in sample:
    if inst[6] == 1:
        satisfies_precondition.append(inst)

satisfies_condition = []

for inst in satisfies_precondition:
    if inst[7] == 1:
        satisfies_condition.append(inst)

print len(satisfies_condition), len(satisfies_precondition)
print len(satisfies_condition) / len(satisfies_precondition)