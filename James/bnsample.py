from __future__ import division
from datetime import datetime
from bnbayesfit import bnbayesfit
import os

struct_fp = os.path.join(os.path.dirname(__file__), 'bnstruct.bn')
data_fp = os.path.join(os.path.dirname(__file__), 'bndata.csv')

def bnsample(fittedbn,nsamples):
	pass
	
if __name__ == "__main__":
	print datetime.now()
	fittedbn = bnbayesfit(struct_fp, data_fp)
	print bnsample(fittedbn, 100)
	print datetime.now()