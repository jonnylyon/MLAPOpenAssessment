from reglinear import reglinear
import os

#features = ['a','b','c','d','e','f','g','h']
features = ['g']

fp = os.path.join(os.path.dirname(__file__), 'stock_price.csv')

for i, feature in enumerate(features):
    power = 1
    inclusion_list = [0,0,0,0,0,0,0,0]
    inclusion_list[i] = power
    
    result = reglinear(fp, inclusion_list)
    
    fp_out = os.path.join(os.path.dirname(__file__),'autorun', '%s%s.txt' % (feature,str(power)))
    
    f = open(fp_out,'w')
    f.write(str(result))
    f.close()