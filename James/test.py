from jc_bayesian import bnbayesfit, bnsample
import os

struct_fp = os.path.join(os.path.dirname(__file__), 'bnstruct.bn')
data_fp = os.path.join(os.path.dirname(__file__), 'bndata.csv')

fittedbn = bnbayesfit(struct_fp, data_fp)
sample = bnsample(fittedbn, 20)

print sample