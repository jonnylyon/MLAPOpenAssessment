from sm_regression import linear, reglinear, logistic, reglogistic
import os

fp = os.path.join(os.path.dirname(__file__), 'stock_price.csv')

print reglogistic(fp)