from sm_regression import linear, reglinear
import os

fp = os.path.join(os.path.dirname(__file__), 'stock_price.csv')

print reglinear(fp)