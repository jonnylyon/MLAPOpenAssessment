from reglogistic import reglogistic
import os

features = ['h']

fp = os.path.join(os.path.dirname(__file__), 'stock_price.csv')

for i, feature in enumerate(features):
    power = 2
    inclusion_list = [0,0,0,0,0,0,0,0]
    inclusion_list[i] = power
    
    result = reglogistic(fp, inclusion_list)
    
    fp_out = os.path.join(os.path.dirname(__file__),'autorun', 'log%s%s.txt' % (feature,str(power)))
    
    f = open(fp_out,'w')
    f.write(str(result))
    f.close()