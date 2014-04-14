import numpy as np
import reglinear

from numpy import matrix, dot, transpose, nditer

def sign(THETA):
	#return divide(m,abs(m)) # doesn't handle zeros well (divide-by-zero)

	out = [0 if theta == 0 else theta / abs(theta) for theta in nditer(THETA)]
	return transpose(matrix(out))

X = matrix([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])

Y = transpose(matrix([5,10,15]))

T = transpose(matrix([0,1,-1,2,-2]))

lamb = 0.5

data = [[0, float(Y[i]), X[i,:]] for i in range(len(X))]

print reglinear.gradient(T.flatten(), data, lamb, sign)

sign_part = (1 - lamb) * sign(T)

# option 1
summation = 0

for i in range(len(X)):
	summation += dot(X[i],Y[i] - dot(T,X[i]))

opt1_ = (-2*lamb*summation)
opt1 = opt1_ + sign_part

#option 2
opt2_ = (-2 * lamb * dot(transpose(X),Y - dot(X,T)))
opt2 = opt2_ + sign_part

#print sign_part
print
#print opt1_
#print opt1
#print
#print opt2_
print opt2