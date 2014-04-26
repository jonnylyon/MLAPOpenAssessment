from __future__ import division

from task2 import log_sum_exp, log_likelihood, prob_a_given_X, gradient, unflatten_theta
from math import log, exp

from numpy import matrix, dot, zeros, reshape, nditer, ndarray

def test_prob_a_given_x_t(a, x, theta):
	num = exp(dot(x, theta[:,a]))
	den = 0.0
	for i in range(theta.shape[1]):
		den += exp(dot(x, theta[:,i]))
	return num / den

def test_l_s_e(XTHETA):
	total = 0.0
	for i in range(5):
		total += exp(XTHETA[:,i])
	return log(total)

def test_grad(theta, data):
	grad = zeros(theta.shape)
	
	for a in range(5):
		for row in data:
			x = row[4]
			c = row[2]
			I = 1 if a == c else 0
			delta = dot(x, I - test_prob_a_given_x_t(a, x, theta))
			delta = [float(elem) for elem in nditer(delta)]
			for i in range(3):
				grad[i,a] += delta[i]
	
	return grad

def test_log_likelihood(theta, data):
	result = 1.0
	for row in data:
		x = row[4]
		c = row[2]
		num = exp(dot(x, theta[:,c]))
		den = 0.0
		for i in range(theta.shape[1]):
			den += exp(dot(x, theta[:,i]))
		result *= num / den
	return log(result)

a_theta = [[1,2,3,19,20],[4,5,6,21,22],[7,8,9,23,24]]
theta = matrix(a_theta)

d1 = matrix([0.10,0.11,0.12])
d2 = matrix([0.13,0.14,0.15])
d3 = matrix([0.16,0.17,0.18])

e1 = [0,0,0,0,d1]
e2 = [0,0,1,0,d2]
e3 = [0,0,2,0,d3]

data = [e1, e2, e3]

print "LSE (exp, act):", test_l_s_e(dot(d1,theta)), log_sum_exp(dot(d1,theta))
print "Log likelihoods (exp, act):", test_log_likelihood(theta,data), log_likelihood(theta,data)
print "P(c=0|X,theta) (exp, act):", test_prob_a_given_x_t(0, d1, theta), prob_a_given_X(theta, d1, 0)
print "Gradient (exp):"
print test_grad(theta, data)
print "Gradient (act):"
print gradient(theta, data)